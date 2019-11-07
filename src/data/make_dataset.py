# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import pickle


def load_raw_data(input_filepath):
    """
    Loads data from .csv files and performs necessary joins. Function returns
    the training as well as the test set.
    """

    click.echo("Loading csv files to memory...")
    building_df = pd.read_csv(input_filepath + "/building_metadata.csv")
    weather_train = pd.read_csv(input_filepath + "/weather_train.csv")
    train = pd.read_csv(input_filepath + "/train.csv")
    weather_test = pd.read_csv(input_filepath + "/weather_test.csv")
    test = pd.read_csv(input_filepath + "/test.csv")

    click.echo("Performing join operations...")
    train = train.merge(building_df, left_on="building_id",
                        right_on="building_id", how="left")
    train = train.merge(weather_train, left_on=["site_id", "timestamp"],
                        right_on=["site_id", "timestamp"])
    test = test.merge(building_df, left_on="building_id",
                      right_on="building_id", how="left")
    test = test.merge(weather_test, left_on=["site_id", "timestamp"],
                      right_on=["site_id", "timestamp"])
    click.echo("Load successful!")
    return train, test


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df_train, df_test = load_raw_data(input_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
