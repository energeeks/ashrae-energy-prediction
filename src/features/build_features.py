import os
import yaml
import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.timer import timer


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from
        (../interim) into data which is ready for usage in ML models
        (saved in ../processed).
    """
    with open("src/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    with timer("Loading interim data"):
        train_df, test_df = load_interim_data(input_filepath)

    with timer("Encoding categorical features"):
        train_df = encode_categorical_data(train_df)
        test_df = encode_categorical_data(test_df)

    with timer("Encoding timestamp features"):
        train_df = encode_timestamp(train_df, circular=cfg["circular_timestamp_encoding"])
        test_df = encode_timestamp(test_df, circular=cfg["circular_timestamp_encoding"])

    if cfg["log_transform_square_feet"]:
        with timer("Taking the log of selected features"):
            train_df["square_feet"] = np.log1p(train_df["square_feet"])
            test_df["square_feet"] = np.log1p(test_df["square_feet"])

    with timer("Calculating age of buildings"):
        train_df = calculate_age_of_building(train_df)
        test_df = calculate_age_of_building(test_df)

    if cfg["encode_wind_direction"]:
        with timer("Encoding wind_direction features"):
            train_df = encode_wind_direction(train_df)
            test_df = encode_wind_direction(test_df)

    if cfg["fill_na_with_zero"]:
        train_df.fillna(0)
        test_df.fillna(0)

    if cfg["add_lag_features"]:
        with timer("Adding Lag Features"):
            train_df = add_lag_features(train_df, cfg["lag_columns"], cfg["lag_windows"])
            test_df = add_lag_features(test_df, cfg["lag_columns"], cfg["lag_windows"])

    if cfg["exclude_faulty_rows"]:
        with timer("Exclude faulty data and outliers"):
            train_df = exclude_faulty_readings(train_df)

    if cfg["add_leaks_to_train"]:
        with timer("Adding Leak Label to training set"):
            train_df = add_leaked_data(train_df, test_df)

    with timer("Sort training set"):
        train_df.sort_values("timestamp", inplace=True)
        train_df.reset_index(drop=True, inplace=True)

    with timer("Dropping specified columns"):
        train_df = drop_columns(train_df, cfg["drop"])
        test_df = drop_columns(test_df, cfg["drop"])

    with timer("Save processed data"):
        save_processed_data(output_filepath, train_df, test_df)


def load_interim_data(input_filepath):
    """
T    Loads interim data which already is preserved as python object due to
    previous processing steps
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")
    test_df = pd.read_pickle(input_filepath + "/test_data.pkl")
    return train_df, test_df


def encode_categorical_data(data_frame):
    """
    Sets a fitting format for categorical data
    """
    # return pd.get_dummies(data_frame, columns=["meter", "primary_use"])
    data_frame["primary_use"] = LabelEncoder().fit_transform(data_frame["primary_use"])
    data_frame["primary_use"] = pd.Categorical(data_frame["primary_use"])
    data_frame["building_id"] = pd.Categorical(data_frame["building_id"])
    data_frame["site_id"] = pd.Categorical(data_frame["site_id"])
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


def calculate_age_of_building(data_frame):
    data_frame["year_built"] = 2019 - data_frame["year_built"]
    return data_frame


def add_lag_features(data_frame, cols, windows):
    for col in cols:
        for window in windows:
            data_frame["{}_{}_lag".format(col, window)] = data_frame \
                .groupby(["site_id", "building_id", "meter"])[col] \
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
    data_frame.loc[data_frame["wind_speed"] == 0, ["wind_direction_sin", "wind_direction_cos"]] = 0
    return data_frame


def add_leaked_data(train_df, test_df):
    leaked_df = pd.read_feather("data/leak/leak.feather")
    leaked_df.loc[leaked_df["meter_reading"] < 0, "meter_reading"] = 0
    leaked_df = leaked_df[leaked_df["building_id"] != 245]
    leaked_df["timestamp"] = leaked_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    test_leak_df = test_df.copy(deep=True)
    test_leak_df = test_leak_df.merge(leaked_df, left_on=["building_id", "meter", "timestamp"],
                                      right_on=["building_id", "meter", "timestamp"], how="left")
    test_leak_df.dropna(subset=["meter_reading"], inplace=True)

    return pd.concat([train_df, test_leak_df])


def drop_columns(data_frame, drop):
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
