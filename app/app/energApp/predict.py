import numpy as np
import pandas as pd
from flask import current_app
from meteocalc import feels_like
from sklearn.preprocessing import LabelEncoder

from .weather import get_forecast, parse_request


def predict_energy_consumption(buildings):
    forecasts = [forecast_for_building(building) for i, building in buildings.iterrows()]
    df = pd.concat(forecasts)
    df.drop(columns="id", inplace=True)
    df = buildings.merge(df, left_on="id", right_on="building_id")
    df["meter"] = 0
    df["floor_count"] = df["floorcount"]
    df["air_temperature"] = df["temp"]
    df["relative_humidity"] = df["humidity"]
    df["dew_temperature"] = df["air_temperature"] - ((100 - df["relative_humidity"]) / 5)
    df["precip_depth_1_hr"] = np.nan
    df["timestamp"] = pd.to_datetime(df["date"])
    df["wind_direction"] = df["deg"]
    df["wind_speed"] = df["speed"]

    df.drop(columns=["id", "name", "floorcount", "latitude", "longitude", "user_id", "temp", "feels_like", "temp_min",
                     "temp_max", "pressure", "sea_level", "grnd_level", "humidity", "temp_kf", "main", "description",
                     "icon", "speed", "deg", "date"], inplace=True)

    df = create_feels_like(df)

    df = encode_timestamp(df)
    df["area_per_floor"] = df["square_feet"] / df["floor_count"]
    df["square_feet"] = np.log(df["square_feet"])
    df["area_per_floor"] = np.log(df["area_per_floor"])
    df["outlier_square_feet"] = label_outlier("square_feet", df)
    df["outlier_area_per_floor"] = label_outlier("area_per_floor", df)
    df = calculate_age_of_building(df)
    df = add_lag_features(df, ["air_temperature", "dew_temperature", "cloud_coverage"], [6, 24])

    df_temp = df.copy(deep=True)
    for i in range(1, 4):
        df_temp["meter"] += 1
        df = pd.concat([df, df_temp])
    del df_temp

    df = encode_categorical_data(df)

    building_ids = df["building_id"]
    timestamps = df["timestamp"]
    df.drop(columns=["timestamp", "month", "wind_direction", "wind_speed", "building_id"], inplace=True)

    lgbm_model = current_app.config['MODEL']
    predictions = pd.DataFrame({"reading": np.expm1(lgbm_model.predict(df)),
                                "building_id": building_ids,
                                "meter": df["meter"],
                                "timestamp": timestamps})
    return predictions


def forecast_for_building(building):
    response = get_forecast(building["latitude"], building["longitude"])
    result = parse_request(response)
    result["building_id"] = building["id"]
    return result


def create_feels_like(df):
    """
    Creates a feels-like temperature feature for the dataframe.
    :param df: weather data frame.
    :return: Dataframe with added feature
    """
    df["air_temp_f"] = df["air_temperature"] * 9 / 5. + 32
    df["feels_like_temp"] = df.apply(lambda x: feels_like_custom(x), axis=1)
    return df


def feels_like_custom(row):
    """
    Computes feels like feature for an entry from the dataframe
    :param row: entry from the weather dataframe
    :return: feels like value for the entry
    """
    temperature = row["air_temp_f"]
    wind_speed = row["wind_speed"]
    humidity = row["relative_humidity"]
    fl = feels_like(temperature, wind_speed, humidity)
    out = fl.c
    return out


def encode_categorical_data(data_frame):
    """
    Sets a fitting format for categorical data.
    """
    # return pd.get_dummies(data_frame, columns=["meter", "primary_use"])
    data_frame["primary_use"] = LabelEncoder().fit_transform(data_frame["primary_use"])
    data_frame["primary_use"] = pd.Categorical(data_frame["primary_use"])
    data_frame["meter"] = pd.Categorical(data_frame["meter"])
    return data_frame


def encode_timestamp(data_frame, circular=False):
    """
    Extracts time based features out of the timestamp column. In particular the
    time of the day, weekday and day of the year were being chosen. Due to the
    repetitive nature of time features a cyclic encoding can been chosen.
    """
    timestamp = data_frame["timestamp"]
    if circular:
        timestamp_seconds_of_day = (timestamp.dt.hour * 60 + timestamp.dt.minute) * 60 + timestamp.dt.second
        data_frame["timeofday_sin"] = np.sin(2 * np.pi * timestamp_seconds_of_day / 86400)
        data_frame["timeofday_cos"] = np.cos(2 * np.pi * timestamp_seconds_of_day / 86400)
        data_frame["dayofweek_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
        data_frame["dayofweek_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)
        data_frame["dayofyear_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofyear / 366)
        data_frame["dayofyear_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofyear / 366)
    else:
        data_frame["hour"] = pd.Categorical(timestamp.dt.hour)
        data_frame["weekday"] = pd.Categorical(timestamp.dt.dayofweek)
        data_frame["month"] = pd.Categorical(timestamp.dt.month)
    return data_frame


def label_outlier(variable, df):
    """
    Flags outliers contained in the dataframe
    :param variable:
    :param df:
    :return: true for each outlier present
    """
    var = df[variable]
    mn = np.mean(var)
    std = np.std(var)
    lower = mn - 2.5 * std
    upper = mn + 2.5 * std
    is_outlier = (var < lower) | (var > upper)
    return is_outlier


def calculate_age_of_building(data_frame):
    """
    Transforms year_built feature in building_metadata.cvs into age.
    :param data_frame:
    :return: dataframe with transformed feature
    """
    data_frame["year_built"] = 2019 - data_frame["year_built"]
    return data_frame


def add_lag_features(data_frame, cols, windows):
    for col in cols:
        for window in windows:
            data_frame["{}_{}_lag".format(col, window)] = data_frame \
                .groupby(["building_id", "meter"])[col] \
                .rolling(window, center=False) \
                .mean().reset_index(drop=True)
    return data_frame

