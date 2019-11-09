# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import os


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


def load_raw_data(input_filepath):
    """
    Loads data from .csv files and performs necessary joins. Function returns
    the training as well as the test set.
    """

    click.echo("Loading csv files to memory...")
    building_df = pd.read_csv(input_filepath + "/building_metadata.csv")
    weather_train = pd.read_csv(input_filepath + "/weather_train.csv")
    train_df = pd.read_csv(input_filepath + "/train.csv")
    weather_test = pd.read_csv(input_filepath + "/weather_test.csv")
    test_df = pd.read_csv(input_filepath + "/test.csv")

    click.echo("Performing join operations...")
    train_df = train_df.merge(building_df, left_on="building_id",
                              right_on="building_id", how="left")
    train_df = train_df.merge(weather_train, left_on=["site_id", "timestamp"],
                              right_on=["site_id", "timestamp"])
    test_df = test_df.merge(building_df, left_on="building_id",
                            right_on="building_id", how="left")
    test_df = test_df.merge(weather_test, left_on=["site_id", "timestamp"],
                            right_on=["site_id", "timestamp"])

    click.echo("Load successful!")
    return train_df, test_df


def save_joined_data(train_df, test_df, output_filepath):
    """
    Takes the two joined dataframes and stores them for further engineering
    """
    os.makedirs(output_filepath, exist_ok=True)
    train_df.to_pickle(output_filepath + "/train_data.pkl")
    test_df.to_pickle(output_filepath + "/test_data.pkl")
    click.echo("Data successfully saved in folder: " + output_filepath)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready for feature engineering (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    click.echo("Starting data loader...")
    train_df, test_df = load_raw_data(input_filepath)

    click.echo("Reducing memory allocation of dataframes...")
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)

    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    # <TODO>
    # FUNCTIONS FOR CLEANSING THE DATA COME HERE
    # --> LOGIC CLEANSING
    # --> HANDLE NANS

    click.echo("Saving cleansed data...")
    save_joined_data(train_df, test_df, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
