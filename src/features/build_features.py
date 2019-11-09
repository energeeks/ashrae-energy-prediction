import os

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from
        (../interim) into data which is ready for usage in ML models
        (saved in ../processed).
    """
    # <TODO>
    # ALL FEATURE ENGINEERING GOES IN HERE
    # OUTPUT COULD BE TWO SETS (ONE FOR BOOSTING, OTHER FOR NN)

    click.echo("Loading interim data...")
    train_df, test_df = load_interim_data(input_filepath)

    click.echo("Encoding timestamp features...")
    train_df = encode_timestamp(train_df)
    test_df = encode_timestamp(test_df)

    click.echo("Encoding wind_direction features...")
    train_df = encode_wind_direction(train_df)
    test_df = encode_wind_direction(test_df)

    click.echo("Encoding categorical features...")
    train_df = encode_categorical_data(train_df)
    test_df = encode_categorical_data(test_df)

    click.echo("Remove nas...")
    train_df.fillna(0)
    test_df.fillna(0)

    click.echo("Save processed data...")
    save_processed_data(output_filepath, train_df, test_df)


def load_interim_data(input_filepath):
    """
    Loads train and test interim pickle data
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")
    test_df = pd.read_pickle(input_filepath + "/test_data.pkl")
    return train_df, test_df


def encode_timestamp(data_frame):
    """
    Encode timeofday, dayofweek and dayofyear with a cyclic encoding
    """
    timestamp = data_frame["timestamp"]
    timestamp_seconds_of_day = (timestamp.dt.hour * 60 + timestamp.dt.minute) * 60 + timestamp.dt.second
    data_frame["timeofday_sin"] = np.sin(2 * np.pi * timestamp_seconds_of_day / 86400)
    data_frame["timeofday_cos"] = np.cos(2 * np.pi * timestamp_seconds_of_day / 86400)
    data_frame["dayofweek_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)
    data_frame["dayofweek_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)
    data_frame["dayofyear_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofyear / 366)
    data_frame["dayofyear_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofyear / 366)
    data_frame.drop(columns=["timestamp"])
    return data_frame


def encode_wind_direction(data_frame):
    """
    Encode the wind_direction using a cyclic encoding.
    If there is no wind_direction, or the wind_speed is zero the points are encoded as the origin.
    """
    data_frame["wind_direction_sin"] = np.sin(2 * np.pi * data_frame["wind_direction"] / 360)
    data_frame["wind_direction_cos"] = np.cos(2 * np.pi * data_frame["wind_direction"] / 360)
    data_frame.loc[data_frame["wind_direction"].isna(), ["wind_direction_sin", "wind_direction_cos"]] = 0
    data_frame.loc[data_frame["wind_speed"] == 0, ["wind_direction_sin", "wind_direction_cos"]] = 0
    data_frame.drop(columns=["wind_direction"])
    return data_frame


def encode_categorical_data(data_frame):
    """
    Encodes categorical data using one hot encoding
    """
    return pd.get_dummies(data_frame, columns=["meter", "primary_use"])


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
