import numpy as np
import pandas as pd

from src.timer import timer


def load_processed_training_data(input_filepath, columns):
    """
    Loads processed data and returns a df with distinguished label column.
    :param input_filepath: Directory that contains the processed data.
    :param columns: The list of columns to load.
    :return Tuple with the Training Data and a vector with the matching labels.
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")
    label = np.log1p(train_df["meter_reading"])

    with timer("Dropping unnecessary columns"):
        train_df = drop_unnecessary_columns(train_df, columns)

    return train_df, label


def load_processed_test_data(input_filepath, columns):
    """
    Loads processed data and returns a df with distinguished row_id column.
    :param input_filepath: Directory that contains the processed data.
    :param columns: The list of columns to load.
    :return Tuple with the Training Data and a vector with the matching labels.
    """
    test_df = pd.read_pickle(input_filepath + "/test_data.pkl")
    row_ids = test_df["row_id"]

    with timer("Dropping unnecessary columns"):
        test_df = drop_unnecessary_columns(test_df, columns)

    return test_df, row_ids


def drop_unnecessary_columns(data_frame, columns):
    """
    Keeps only selected columns from data_frame
    :param data_frame:
    :param columns: the columns to keep
    :return: data_frame with dropped columns
    """
    missing_columns = [c for c in columns if c not in data_frame.columns]
    if missing_columns:
        raise ValueError("data frame is missing columns specified in config: " + missing_columns)

    unnecessary_cols = [c for c in data_frame.columns if c not in columns]
    data_frame.drop(columns=unnecessary_cols, inplace=True)
    return data_frame
