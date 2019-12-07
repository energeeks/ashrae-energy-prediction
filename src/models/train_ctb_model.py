import os
import random
import yaml
import click
import catboost as ctb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.timer import timer


@click.command()
@click.argument('mode')
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(mode, input_filepath, output_filepath):
    """
    Collects prepared data and starts training an CatBoost model. Parameters
    can be specified by editing src/config.yml.
    """
    random.seed(1337)
    with timer("Loading processed training data"):
        train_df, label = load_processed_training_data(input_filepath)

    ###########################################################################
    # DEFINE PARAMETERS FOR THE CATBOOST MODEL                                     #
    ###########################################################################
    with open("src/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    params = cfg["ctb_params"]
    early_stopping_rounds = cfg["ctb_early_stopping_rounds"]
    splits = cfg["ctb_splits_for_cv"]
    verbose_eval = cfg["ctb_verbose_eval"]
    ###########################################################################

    if mode == "cv":
        start_cv_run(train_df, label, params, splits, verbose_eval,
                     early_stopping_rounds, output_filepath)
    else:
        raise ValueError("Choose a valid mode: 'cv'")


def load_processed_training_data(input_filepath):
    """
    Loads processed data and returns a df with distinguished label
    column.
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")

    label = np.log1p(train_df["meter_reading"])
    del train_df["meter_reading"]

    return train_df, label


def start_cv_run(train_df, label, params, splits, verbose_eval,
                 early_stopping_rounds, output_filepath):
    """
    Starts a Cross Validation Run with the parameters provided.
    Scores will be documented and models will be saved.
    """
    output_filepath = output_filepath + "_cv"
    cv_results = []
    with timer("Performing " + str(splits) + " fold cross-validation"):
        kf = KFold(n_splits=splits, shuffle=False, random_state=1337)
        for i, (train_index, test_index) in enumerate(kf.split(train_df, label)):
            with timer("~~~~ Fold %d of %d ~~~~" % (i + 1, splits)):
                x_train, x_valid = train_df.iloc[train_index], train_df.iloc[test_index]
                y_train, y_valid = label[train_index], label[test_index]
                cat_features = list(x_train.select_dtypes(include=['category']).columns)

                ctb_model = ctb.CatBoostRegressor(**params)
                ctb_model.fit(x_train,
                              y_train,
                              cat_features=cat_features,
                              eval_set=(x_valid, y_valid),
                              verbose_eval=verbose_eval,
                              early_stopping_rounds=early_stopping_rounds)
                save_model(output_filepath, ctb_model)


def save_model(output_filepath, model):
    """
    Saves the trained model in /models/ctb
    """
    os.makedirs(output_filepath, exist_ok=True)
    files_in_dir = os.listdir(output_filepath)
    max_version = max([int(file[:4]) for file in files_in_dir], default=0)
    new_version = str(max_version + 1).zfill(4)
    model.save_model(output_filepath + "/" + new_version + ".txt")
    click.echo("Model successfully saved in folder: " + output_filepath)


if __name__ == '__main__':
    main()
