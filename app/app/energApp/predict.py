import numpy as np
import pandas as pd
import requests
from src.features.build_features import build_features
from meteocalc import feels_like

from .weather import get_forecast, parse_request


def predict_energy_consumption(buildings):
    """
    Predicts energy consumption with a provided list of buildings.
    The model is being served as a rest endpoint.
    :param buildings: List of buildings for which the prediction should be done.
    :return: Data frame with the predicted readings.
    """
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

    df_temp = df.copy(deep=True)
    for i in range(1, 4):
        df_temp["meter"] += 1
        df = pd.concat([df, df_temp])
    del df_temp

    cfg = {
        'circular_timestamp_encoding': False,
        'log_transform_square_feet': True,
        'log_transform_area_per_floor': True,
        'label_square_feet_outlier': True,
        'label_area_per_floor_outlier': True,
        'encode_wind_direction': False,
        'include_feels_like': True,
        'fill_na_with_zero': False,
        'add_lag_features': True,
        'lag_columns': ['air_temperature', 'dew_temperature', 'cloud_coverage'],
        'lag_windows': [6, 24],
    }
    [df] = build_features(df, cfg=cfg)

    df.reset_index(inplace=True, drop=True)
    building_ids = df["building_id"]
    timestamps = df["timestamp"]
    df.drop(columns=["timestamp", "month", "wind_direction", "wind_speed", "building_id"], inplace=True)

    model_endpoint = "http://model:5001/predict"
    data = df.to_json()
    response = requests.get(model_endpoint, json=data).json()

    predictions = pd.DataFrame({"reading": response["prediction"],
                                "building_id": building_ids,
                                "meter": df["meter"],
                                "timestamp": timestamps,
                                "air_temperature": df["air_temperature"]})
    return predictions


def forecast_for_building(building):
    """
    Based on longitude and latitude a weather prediction is fetched from the API.
    :param building: A particular building
    :return: weather forecast for the specific building.
    """
    response = get_forecast(building["latitude"], building["longitude"])
    result = parse_request(response)
    result["building_id"] = building["id"]
    return result
