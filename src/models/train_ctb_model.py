import os

import catboost as ctb
import click
from sklearn.model_selection import KFold

from src.models.model_utils import load_processed_training_data
from src.timer import timer


def train_ctb_model(mode, input_filepath, output_filepath, cfg):
    """
    Collects prepared data and starts training an CatBoost model.
    :param mode: Specifies mode to run. Now only cv (cross validation) is supported.
    :param input_filepath: Directory that contains the processed data.
    :param output_filepath: Directory that will contain the trained models.
    :param cfg: Config read from src/config.yml.
    """
    with timer("Loading processed training data"):
        train_df, label = load_processed_training_data(input_filepath, cfg["columns"])

    ###########################################################################
    # DEFINE PARAMETERS FOR THE LGBM MODEL                                     #
    ###########################################################################
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


def start_cv_run(train_df, label, params, splits, verbose_eval,
                 early_stopping_rounds, output_filepath):
    """
    Starts a Cross Validation Run with the parameters provided.
    Scores will be documented and models will be saved.
    :param train_df: DataFrame which contains the training data.
    :param label: A vector which contains the labels of the training data.
    :param params: Dictionary with the model parameters
    :param splits: Integer describing the number of folds / splitting fraction.
    :param verbose_eval: The interval where training information is printed
    to console.
    :param early_stopping_rounds: If no improvement of the validation score in
    n rounds occur, the training will be stopped.
    :param output_filepath: Directory that will contain the trained models.
    """
    output_filepath = output_filepath + "_cv"
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
    Saves the trained model.
    :param output_filepath: Directory that will contain the trained models.
    :param model: Trained model that needs to be saved to disc.
    """
    os.makedirs(output_filepath, exist_ok=True)
    files_in_dir = os.listdir(output_filepath)
    max_version = max([int(file[:4]) for file in files_in_dir], default=0)
    new_version = str(max_version + 1).zfill(4)
    model.save_model(output_filepath + "/" + new_version + ".txt")
    click.echo("Model successfully saved in folder: " + output_filepath)
