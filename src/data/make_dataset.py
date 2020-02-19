# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytz
import yaml
from dotenv import find_dotenv, load_dotenv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from src.timer import timer


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def click_main(data_dir, output_dir):
    main(data_dir, output_dir)


def main(data_dir, output_dir):
    """
    Runs data processing scripts to turn raw data (data_dir/raw) and external
    data (data_dir/external) into cleaned data ready for feature engineering
    (saved in output_dir).
    :param data_dir: Directory that contains the data
    :param output_dir: Directory where results will be saved in.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    with open("src/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    with timer("Loading data"):
        train_df = load_main_csv(data_dir + "/raw/train.csv")
        test_df = load_main_csv(data_dir + "/raw/test.csv")
        weather_train_df = load_weather_csv(data_dir + "/raw/weather_train.csv")
        weather_test_df = load_weather_csv(data_dir + "/raw/weather_test.csv")
        building_df = load_building_csv(data_dir + "/raw/building_metadata.csv")
        site_df = load_site_csv(data_dir + "/external/site_info.csv")

    with timer("Merging main and building"):
        train_df = train_df.merge(building_df, on="building_id", how="left")
        test_df = test_df.merge(building_df, on="building_id", how="left")

    if cfg["impute_weather_data"]:
        with timer("Impute missing weather data"):
            weather_train_df = impute_weather_data(weather_train_df)
            weather_test_df = impute_weather_data(weather_test_df)

    with timer("Merging weather and site"):
        weather_train_df = weather_train_df.merge(site_df, on="site_id", how="left")
        weather_test_df = weather_test_df.merge(site_df, on="site_id", how="left")

    if cfg["localize_timestamps"]:
        with timer("Localizing weather timestamp"):
            weather_train_df = localize_weather_timestamp(weather_train_df)
            weather_test_df = localize_weather_timestamp(weather_test_df)

    with timer("Merging main and weather"):
        train_df = train_df.merge(weather_train_df, on=["site_id", "timestamp"], how="left")
        test_df = test_df.merge(weather_test_df, on=["site_id", "timestamp"], how="left")

    with timer("Saving cleansed data"):
        save_joined_data(train_df, test_df, output_dir)


def load_main_csv(csv):
    """
    Reads, parses and converts the data contained into the main dataframe
    (data_dir/raw/train.csv), which include the meter readings per building.
    Each feature is converted accordingly before the dataframe is read.
    :param csv: train.csv retrieved from kaggle
    :return: Parsed and converted dataframe
    """
    column_types = {
        "building_id": np.uint16,
        "meter": np.uint8,
        "timestamp": np.datetime64,
        "meter_reading": np.float32,
    }
    dtype, parse_dates, converters = split_column_types(column_types)
    return pd.read_csv(csv, dtype=dtype, parse_dates=parse_dates, converters=converters)


def load_weather_csv(csv):
    """
    Reads, parses and converts the data contained into the weather dataframe (data_dir/raw/weather_train.csv).
    Each feature is converted accordingly before the dataframe is read.
    :param csv: weather_train.csv retrieved from kaggle
    :return: Parsed and converted weather dataframe
    """
    column_types = {
        "site_id": np.uint8,
        "timestamp": np.datetime64,
        "air_temperature": np.float16,
        "cloud_coverage": np.float16,
        "dew_temperature": np.float16,
        "precip_depth_1_hr": np.float16,
        "sea_level_pressure": np.float16,
        "wind_direction": np.float16,
        "wind_speed": np.float16,
    }
    dtype, parse_dates, converters = split_column_types(column_types)
    return pd.read_csv(csv, dtype=dtype, parse_dates=parse_dates, converters=converters)


def load_building_csv(csv):
    """
    Reads, parses and converts the data contained into the building_metadata dataframe
    (data_dir/raw/building_metadata.csv). Each feature is handled and converted accordingly
    before the dataframe is read.
    :param csv: building_metadata.csv retrieved from kaggle
    :return: Parsed and converted building_metadata dataframe
    """
    column_types = {
        "site_id": np.uint8,
        "timezone": pytz.timezone,
        "country_code": np.object,
        "location": np.object,
    }
    dtype, parse_dates, converters = split_column_types(column_types)
    return pd.read_csv(csv, dtype=dtype, parse_dates=parse_dates, converters=converters)


def load_site_csv(csv):
    """
    Reads, parses and converts the data contained into the site_info (data_dir/external/site_info.csv).
    :param csv: site_info.csv retrieved from kaggle
    :return: Parsed and converted site_info dataframe
    """
    column_types = {
        "site_id": np.uint8,
        "timezone": pytz.timezone,
        "country_code": np.object,
        "location": np.object,
    }
    dtype, parse_dates, converters = split_column_types(column_types)
    return pd.read_csv(csv, delimiter=";", dtype=dtype, parse_dates=parse_dates, converters=converters)


def split_column_types(column_types):
    """
    Provided a list of column_types the method will set fitting parameters
    for the csv import
    :param column_types: Dictionary containing column: datatype
    :return: Tuple with the assigned data type and whether to parse the date or not
    """

    def is_parse_date(it):
        return it == np.datetime64

    def is_dtype(it):
        try:
            np.dtype(it)
            return not is_parse_date(it)
        except:
            return False

    def is_converter(it):
        return not is_dtype(it) and not is_parse_date(it)

    dtype = {k: v for k, v in column_types.items() if is_dtype(v)}
    parse_dates = [k for k, v in column_types.items() if is_parse_date(v)]
    converters = {k: v for k, v in column_types.items() if is_converter(v)}
    return dtype, parse_dates, converters


def impute_weather_data(data_frame):
    """
    Imputes missing data from the weather dataframe using iterative imputer
    :param data_frame: weather dataframe
    :return: dataframe with imputed values
    """
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])
    min_date = data_frame["timestamp"].dropna().min()
    max_date = data_frame["timestamp"].dropna().max()
    date_range = pd.date_range(start=min_date, end=max_date, freq="1H")
    date_range = pd.to_datetime(date_range)
    date_range = pd.DataFrame({"timestamp": date_range})
    weather_imputed = pd.DataFrame(columns=["timestamp", "site_id"])

    # Create perfect timeline without missing hours
    for site in data_frame["site_id"].unique():
        date_range["site_id"] = site
        weather_imputed = weather_imputed.append(date_range)

    # Join with existing weather data
    weather_imputed = weather_imputed.merge(data_frame, left_on=["site_id", "timestamp"],
                                            right_on=["site_id", "timestamp"],
                                            how="left")

    # Create new temporal features for better imputation
    weather_imputed["hour"] = pd.Categorical(weather_imputed["timestamp"].dt.hour)
    weather_imputed["weekday"] = pd.Categorical(weather_imputed["timestamp"].dt.dayofweek)
    weather_imputed["month"] = pd.Categorical(weather_imputed["timestamp"].dt.month)

    # Preserve data_frame data before transforming
    weather_cols = weather_imputed.columns.values
    weather_timestamp = weather_imputed["timestamp"]
    weather_site_ids = weather_imputed["site_id"]

    # Scale data for algorithm
    date_delta = pd.datetime.now() - weather_imputed["timestamp"]
    weather_imputed["timestamp"] = date_delta.dt.total_seconds()
    scaler = StandardScaler()
    weather_imputed = scaler.fit_transform(weather_imputed)

    # Impute missing values
    imputer = IterativeImputer(max_iter=20,
                               initial_strategy="median")
    weather_imputed = imputer.fit_transform(weather_imputed)

    # Rescale
    weather_imputed = scaler.inverse_transform(weather_imputed)

    # Assemble final weather frame
    weather_final = pd.DataFrame(data=weather_imputed, columns=weather_cols)
    weather_final["timestamp"] = weather_timestamp
    weather_final["site_id"] = weather_site_ids
    weather_final = weather_final.drop(columns=["hour", "weekday", "month"], axis=1)

    return weather_final


def localize_weather_timestamp(df):
    """
    Localizes all weather dataframe timestamps, drops unwanted duplicates
    which might have been generated.
    :param df: weather dataframe
    :return: dataframe with localized timestamps
    """
    key = ["site_id", "timestamp"]
    df.sort_values(by=key, inplace=True)  # Sort for drop_duplicates
    df["timestamp"] = df.apply(localize_row_timestamp, axis=1)
    df.drop_duplicates(subset=key, keep="last", inplace=True)  # Because of DST we can have duplicates here
    df.reset_index(drop=True, inplace=True)
    return df


def localize_row_timestamp(row):
    """
    Convert timestamp of an entry to the local timezone
    :param row: row of the weather dataframe
    :return: converted timestamps
    """
    return convert_time_zone(row["timestamp"], to_tz=row["timezone"])


def convert_time_zone(dt, from_tz=pytz.utc, to_tz=pytz.utc):
    """
    Converts timestamps to local timezone
    :param dt: A timestamp
    :param from_tz: Origin timezone
    :param to_tz: Desired timezone
    :return: Dataframe with localized values of timezones
    """
    return dt.tz_localize(from_tz).tz_convert(to_tz).tz_localize(None)


def save_joined_data(train_df, test_df, output_dir):
    """
    Takes the two joined dataframes and stores them for further engineering.
    :param train_df: Data frame containing training data
    :param test_df: Data frame containing test data
    :param output_dir: Directory where the results will be saved in.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_pickle(output_dir + "/train_data.pkl")
    test_df.to_pickle(output_dir + "/test_data.pkl")
    click.echo("Data successfully saved in folder: " + output_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    click_main()
