import os

import click
import math
import numpy as np
import pandas as pd
import yaml
from meteocalc import feels_like

from src.timer import timer


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from
        (../interim) into data which is ready for usage in ML models
        (saved in ../processed).
        :param input_filepath: Directory that contains the interim data
        :param output_filepath: Directory where processed results will be saved in.
    """
    with open("src/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    with timer("Loading interim data"):
        train_df, test_df = load_interim_data(input_filepath)

    train_df, test_df = build_features(train_df, test_df, cfg=cfg)

    if cfg["exclude_faulty_rows"]:
        with timer("Exclude faulty data and outliers"):
            train_df = exclude_faulty_readings(train_df)

    if cfg["add_leaks_to_train"]:
        with timer("Adding Leak Label to training set"):
            train_df = add_leaked_data(train_df, test_df)

    with timer("Sort training set"):
        train_df.sort_values("timestamp", inplace=True)
        train_df.reset_index(drop=True, inplace=True)
       
    if cfg["drop_buildings_with_leaks"]:
        with timer("Drop buildings with leaks from train set"):
            buildings_with_leaks = [131, 163, 166, 168, 174, 179, 201]
            train_df[np.invert(train_df.building_id.isin(buildings_with_leaks))]
            train_df.reset_index(drop = True, inplace = True)

    with timer("Dropping specified columns"):
        train_df = drop_columns(train_df, cfg["drop"])
        test_df = drop_columns(test_df, cfg["drop"])

    with timer("Save processed data"):
        save_processed_data(output_filepath, train_df, test_df)


def load_interim_data(input_filepath):
    """
    Loads interim data which already is preserved as python object due to
    previous processing steps
    :param input_filepath: Directory that contains the interim data
    :return: Tuple containing training and test data
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")
    test_df = pd.read_pickle(input_filepath + "/test_data.pkl")
    return train_df, test_df


def build_features(*dfs, cfg):
    with timer("Encoding categorical features"):
        dfs = [encode_categorical_data(df) for df in dfs]

    with timer("Encoding timestamp features"):
        dfs = [encode_timestamp(df, circular=cfg["circular_timestamp_encoding"]) for df in dfs]

    with timer("Create area per floor feature"):
        dfs = [calculate_area_per_floor(df) for df in dfs]

    if cfg["log_transform_square_feet"]:
        with timer("Taking the log of selected features"):
            dfs = [calculate_square_feet_log(df) for df in dfs]

    if cfg["log_transform_area_per_floor"]:
        with timer("Taking the log of area per floor"):
            dfs = [calculate_area_per_floor_log(df) for df in dfs]

    if cfg["label_square_feet_outlier"]:
        with timer("Create outlier label for square feet"):
            dfs = [label_square_feet_outlier(df) for df in dfs]

    if cfg["label_area_per_floor_outlier"]:
        with timer("Create outlier label for area per floor"):
            dfs = [label_area_per_floor_outlier(df) for df in dfs]

    with timer("Calculating age of buildings"):
        dfs = [calculate_age_of_building(df) for df in dfs]

    if cfg["encode_wind_direction"]:
        with timer("Encoding wind_direction features"):
            dfs = [encode_wind_direction(df) for df in dfs]

    with timer("Calculate relative humidity"):
        dfs = [calculate_relative_humidity(df) for df in dfs]

    if cfg["include_feels_like"]:
        with timer("Create feels_like_temp"):
            dfs = [calculate_feels_like_temp(df) for df in dfs]

    if cfg["fill_na_with_zero"]:
        dfs = [df.fillna(0) for df in dfs]

    if cfg["add_lag_features"]:
        with timer("Adding Lag Features"):
            dfs = [add_lag_features(df, cfg["lag_columns"], cfg["lag_windows"]) for df in dfs]

    return dfs


def encode_categorical_data(data_frame):
    """
    Sets a fitting format for categorical data.
    """
    primary_use_label = {
        'Education': 0,
        'Entertainment/public assembly': 1,
        'Food sales and service': 2,
        'Healthcare': 3,
        'Lodging/residential': 4,
        'Manufacturing/industrial': 5,
        'Office': 6,
        'Other': 7,
        'Parking': 8,
        'Public services': 9,
        'Religious worship': 10,
        'Retail': 11,
        'Services': 12,
        'Technology/science': 13,
        'Utility': 14,
        'Warehouse/storage': 15
    }
    data_frame["primary_use"].replace(primary_use_label, inplace=True)

    columns = ["site_id", "building_id", "meter", "primary_use"]
    for column in columns:
        if column in data_frame.columns:
            data_frame[column] = pd.Categorical(data_frame[column])
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


def calculate_area_per_floor(df):
    df["area_per_floor"] = df["square_feet"] / df["floor_count"]
    return df


def calculate_square_feet_log(df):
    df["square_feet"] = np.log(df["square_feet"])
    return df


def calculate_area_per_floor_log(df):
    df["area_per_floor"] = np.log(df["area_per_floor"])
    return df


def label_square_feet_outlier(df):
    df["outlier_square_feet"] = label_outlier("square_feet", df)
    return df


def label_area_per_floor_outlier(df):
    df["outlier_area_per_floor"] = label_outlier("area_per_floor", df)
    return df


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


def exclude_faulty_readings(data_frame):
    """"
    Cleanses the provided data_frame from faulty readings and/or outlier data.
    Special thanks goes to https://www.kaggle.com/purist1024/ashrae-simple-data
    -cleanup-lb-1-08-no-leaks for providing a detailed guide and identification
    of the problematical rows.
    """
    rows_to_drop = pd.read_csv("data/external/rows_to_drop.csv")
    return data_frame.drop(index=rows_to_drop.iloc[:, 0])


def encode_wind_direction(data_frame):
    """
    Encode the wind_direction using a cyclic encoding.
    If there is no wind_direction or the wind_speed is zero the points are encoded as the origin.
    """
    data_frame["wind_direction_sin"] = np.sin(2 * np.pi * data_frame["wind_direction"] / 360)
    data_frame["wind_direction_cos"] = np.cos(2 * np.pi * data_frame["wind_direction"] / 360)
    data_frame.loc[data_frame["wind_direction"].isna(), ["wind_direction_sin", "wind_direction_cos"]] = 0
    data_frame.loc[data_frame["wind_speed"].isna(), ["wind_direction_sin", "wind_direction_cos"]] = 0
    data_frame.loc[data_frame["wind_speed"] == 0, ["wind_direction_sin", "wind_direction_cos"]] = 0
    return data_frame


def calculate_relative_humidity(df):
    subset = df[["air_temperature", "dew_temperature"]].drop_duplicates()
    subset["relative_humidity"] = subset.apply(
        lambda row: calculate_row_relative_humidity(row["air_temperature"], row["dew_temperature"]), axis=1)
    return df.merge(subset, on=["air_temperature", "dew_temperature"])


def calculate_row_relative_humidity(air_temperature, dew_temperature):
    """
    Computes the relative humidity from air temperature and dew point.
    :param air_temperature: the dry air temperature
    :param dew_temperature: the dew point
    :return: the relative humidity
    """
    positive = {'b': 17.368, 'c': 238.88}
    negative = {'b': 17.966, 'c': 247.15}
    const = positive if air_temperature > 0 else negative
    pa = math.exp(dew_temperature * const['b'] / (const['c'] + dew_temperature))
    rel_humidity = pa * 100. * 1 / math.exp(const['b'] * air_temperature / (const['c'] + air_temperature))
    return rel_humidity


def calculate_feels_like_temp(df):
    """
    Creates a feels-like temperature feature for the dataframe.
    :param df: weather data frame.
    :return: Dataframe with added feature
    """
    subset = df[["air_temperature", "wind_speed", "relative_humidity"]].drop_duplicates()
    subset["feels_like_temp"] = subset.apply(
        lambda row: calculate_row_feels_like_temp(row["air_temperature"], row["wind_speed"], row["relative_humidity"]),
        axis=1)
    return df.merge(subset, on=["air_temperature", "wind_speed", "relative_humidity"])


def calculate_row_feels_like_temp(air_temperature, wind_speed, relative_humidity):
    """
    Computes feels like feature for an entry from the dataframe
    :param air_temperature: air temperature in celsius
    :param wind_speed: wind speed
    :param relative_humidity: relative humidity
    :return: feels like value for the entry
    """
    air_temperature_fahrenheit = air_temperature * 9 / 5 + 32
    fl = feels_like(air_temperature_fahrenheit, wind_speed, relative_humidity)
    out = fl.c
    return out


def add_leaked_data(train_df, test_df):
    """
    Adds the leaked data published in public notebooks in Kaggle's website
    :param train_df:
    :param test_df:
    :return: concatenated dataframe
    """
    leaked_df = pd.read_feather("data/leak/leak.feather")
    leaked_df.loc[leaked_df["meter_reading"] < 0, "meter_reading"] = 0
    leaked_df = leaked_df[leaked_df["building_id"] != 245]

    test_leak_df = test_df.copy(deep=True)
    test_leak_df = test_leak_df.merge(leaked_df, left_on=["building_id", "meter", "timestamp"],
                                      right_on=["building_id", "meter", "timestamp"], how="left")
    test_leak_df.dropna(subset=["meter_reading"], inplace=True)
    del test_leak_df["row_id"]

    return pd.concat([train_df, test_leak_df], sort=False)


def drop_columns(data_frame, drop):
    """
    Drops selected columns from dataframe
    :param data_frame:
    :param drop:
    :return: dataframe with dropped dataframe
    """
    data_frame.drop(columns=drop, inplace=True)
    return data_frame


def save_processed_data(output_filepath, train_df, test_df):
    """
    Saves the processed data
    """
    os.makedirs(output_filepath, exist_ok=True)
    train_df.to_pickle(output_filepath + "/train_data.pkl")
    test_df.to_pickle(output_filepath + "/test_data.pkl")
    click.echo("Data successfully saved in folder: " + output_filepath)


if __name__ == '__main__':
    main()
