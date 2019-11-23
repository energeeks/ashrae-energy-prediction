# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from src.timer import timer
from sklearn.preprocessing import StandardScaler
from fancyimpute import KNN

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready for feature engineering (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    with timer("Loading data"):
        train_df, test_df, weather_train, weather_test = load_raw_data(input_filepath)

    with timer("Impute missing weather data"):
        weather_train = impute_weather_data(weather_train)
        weather_test = impute_weather_data(weather_test)

    with timer("Merge weather data"):
        train_df = merge_weather_data(train_df, weather_train)
        test_df = merge_weather_data(test_df, weather_test)

    with timer("Reducing memory allocation of dataframes"):
        train_df = reduce_mem_usage(train_df)
        test_df = reduce_mem_usage(test_df)

    with timer("Changing column types"):
        train_df = adjust_column_types(train_df)
        test_df = adjust_column_types(test_df)

    # <TODO>
    # FUNCTIONS FOR CLEANSING THE DATA COME HERE
    # --> LOGIC CLEANSING
    # --> HANDLE NANS

    with timer("Saving cleansed data"):
        save_joined_data(train_df, test_df, output_filepath)


def load_raw_data(input_filepath):
    """
    Loads data from .csv files and performs necessary joins. Function returns
    the training as well as the test set.
    """

    with timer("Loading csv files to memory"):
        building_df = pd.read_csv(input_filepath + "/building_metadata.csv")
        weather_train = pd.read_csv(input_filepath + "/weather_train.csv")
        train_df = pd.read_csv(input_filepath + "/train.csv")
        weather_test = pd.read_csv(input_filepath + "/weather_test.csv")
        test_df = pd.read_csv(input_filepath + "/test.csv")

        train_df = train_df.merge(building_df, left_on="building_id", right_on="building_id", how="left")
        test_df = test_df.merge(building_df, left_on="building_id", right_on="building_id", how="left")

    return train_df, test_df, weather_train, weather_test


def impute_weather_data(data_frame):
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
    weather_imputed = merge_weather_data(weather_imputed, data_frame)

    # Preserve data_frame data before transforming
    weather_cols = weather_imputed.columns.values
    weather_timestamp = weather_imputed["timestamp"]
    weather_site_ids = weather_imputed["site_id"]

    # Scale data for KNN
    date_delta = pd.datetime.now() - weather_imputed["timestamp"]
    weather_imputed["timestamp"] = date_delta.dt.total_seconds()
    scaler = StandardScaler()
    weather_imputed = scaler.fit_transform(weather_imputed)

    # Impute missing values
    weather_imputed = KNN(5).fit_transform(weather_imputed)

    # Rescale
    weather_imputed = scaler.inverse_transform(weather_imputed)

    # Assemble final weather frame
    weather_final = pd.DataFrame(data=weather_imputed, columns=weather_cols)
    weather_final["timestamp"] = weather_timestamp
    weather_final["site_id"] = weather_site_ids

    return weather_final


def merge_weather_data(data_frame, weather_df):
    """
    Merges the data_frame with the weather data.
    """
    data_frame = data_frame.merge(weather_df, left_on=["site_id", "timestamp"], right_on=["site_id", "timestamp"],
                                  how="left")
    return data_frame


def reduce_mem_usage(df, verbose=True):
    """
    Takes an dataframe as argument and adjusts the datatypes of the respective
    columns to reduce memory allocation
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min and
                        c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min and
                      c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min and
                      c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min and
                      c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min and
                        c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min and
                      c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    reduced_mem = 100 * (start_mem - end_mem) / start_mem
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
              .format(end_mem, reduced_mem))
    return df


def adjust_column_types(data_frame):
    """
    Takes a data frame and parses certain columns to the desired type.
    """
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])
    return data_frame


def save_joined_data(train_df, test_df, output_filepath):
    """
    Takes the two joined dataframes and stores them for further engineering
    """
    os.makedirs(output_filepath, exist_ok=True)
    train_df.to_pickle(output_filepath + "/train_data.pkl")
    test_df.to_pickle(output_filepath + "/test_data.pkl")
    click.echo("Data successfully saved in folder: " + output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
