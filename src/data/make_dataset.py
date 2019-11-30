# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pytz
from dotenv import find_dotenv, load_dotenv

from src.timer import timer


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def main(data_dir, output_dir):
    """
    Runs data processing scripts to turn raw data (data_dir/raw) and external data (data_dir/external) into cleaned data
    ready for feature engineering (saved in output_dir).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

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

    with timer("Merging weather and site"):
        weather_train_df = weather_train_df.merge(site_df, on="site_id", how="left")
        weather_test_df = weather_test_df.merge(site_df, on="site_id", how="left")

    with timer("Localizing weather timestamp"):
        weather_train_df = localize_timestamp(weather_train_df)
        weather_test_df = localize_timestamp(weather_test_df)

    with timer("Merging main and weather"):
        train_df = train_df.merge(weather_train_df, on=["site_id", "timestamp"], how="left")
        test_df = test_df.merge(weather_test_df, on=["site_id", "timestamp"], how="left")

    with timer("Saving cleansed data"):
        save_joined_data(train_df, test_df, output_dir)


def load_main_csv(csv):
    column_types = {
        "building_id": np.uint16,
        "meter": np.uint8,
        "timestamp": np.datetime64,
        "meter_reading": np.float32,
    }
    dtype, parse_dates, converters = split_column_types(column_types)
    return pd.read_csv(csv, dtype=dtype, parse_dates=parse_dates, converters=converters)


def load_weather_csv(csv):
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
    column_types = {
        "site_id": np.uint8,
        "timezone": pytz.timezone,
        "country_code": np.object,
        "location": np.object,
    }
    dtype, parse_dates, converters = split_column_types(column_types)
    return pd.read_csv(csv, dtype=dtype, parse_dates=parse_dates, converters=converters)


def load_site_csv(csv):
    column_types = {
        "site_id": np.uint8,
        "timezone": pytz.timezone,
        "country_code": np.object,
        "location": np.object,
    }
    dtype, parse_dates, converters = split_column_types(column_types)
    return pd.read_csv(csv, delimiter=";", dtype=dtype, parse_dates=parse_dates, converters=converters)


def split_column_types(column_types):
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


def localize_timestamp(df):
    df.sort_values(by="timestamp", inplace=True)  # Sort for drop_duplicates
    df["timestamp"] = df.apply(localize_row_timestamp, axis=1)
    df.drop_duplicates(subset="timestamp", keep="last", inplace=True)  # Because of DST we can have duplicates here
    df.reset_index(drop=True, inplace=True)
    return df


def localize_row_timestamp(row):
    return convert_time_zone(row["timestamp"], to_tz=row["timezone"])


def convert_time_zone(dt, from_tz=pytz.utc, to_tz=pytz.utc):
    return dt.tz_localize(from_tz).tz_convert(to_tz).tz_localize(None)


def save_joined_data(train_df, test_df, output_dir):
    """
    Takes the two joined dataframes and stores them for further engineering
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

    main()
