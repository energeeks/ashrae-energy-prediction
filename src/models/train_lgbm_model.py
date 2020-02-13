import os
import random
import yaml
import click
import lightgbm as lgb
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
    Collects prepared data and starts training an LightGBM model. Parameters
    can be specified by editing src/config.yml.
    :param mode: Specifies mode to run. Options are full (no validation set,
    single fold), cv (cross validation), by_meter (training by meter type),
    by_building (training by building id).
    :param input_filepath: Directory that contains the processed data.
    :param output_filepath: Directory that will contain the trained models.
    """
    random.seed(1337)
    with timer("Loading processed training data"):
        train_df, label = load_processed_training_data(input_filepath)

    ###########################################################################
    # DEFINE PARAMETERS FOR THE LGBM MODEL                                     #
    ###########################################################################
    with open("src/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    params = cfg["lgbm_params"]
    num_boost_round = cfg["lgbm_num_boost_round"]
    early_stopping_rounds = cfg["lgbm_early_stopping_rounds"]
    splits = cfg["lgbm_splits_for_cv"]
    verbose_eval = cfg["lgbm_verbose_eval"]
    grouped_on_building = cfg["lgbm_cv_grouped_on_building"]
    ###########################################################################

    if mode == "full":
        start_full_training_run(train_df, label, params, verbose_eval,
                                num_boost_round, early_stopping_rounds,
                                output_filepath)
    elif mode == "cv":
        start_cv_run(train_df, label, params, splits, verbose_eval,
                     num_boost_round, early_stopping_rounds, output_filepath)
    elif mode == "by_meter":
        start_full_by_meter_run(train_df, label, params, verbose_eval,
                                num_boost_round, early_stopping_rounds, output_filepath)
    elif mode == "by_building":
        start_full_by_building_run(train_df, label, params, splits, verbose_eval,
                                   num_boost_round, early_stopping_rounds, output_filepath)
    else:
        raise ValueError("Choose a valid mode: 'full', 'cv'")


def load_processed_training_data(input_filepath):
    """
    Loads processed data and returns a df with distinguished label
    column.
    :param input_filepath: Directory that contains the processed data.
    :return Tuple with the Training Data and a vector with the matching labels.
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")

    label = np.log1p(train_df["meter_reading"])
    del train_df["meter_reading"]

    return train_df, label


def start_full_training_run(train_df, label, params, verbose_eval,
                            num_boost_round, early_stopping_rounds, output_filepath):
    """"
    Starts a full training run with the provided parameters.
    :param train_df: DataFrame which contains the training data.
    :param label: A vector which contains the labels of the training data.
    :param params: Dictionary with the model parameters
    :param verbose_eval: The interval where training information is printed
    to console.
    :param num_boost_round: Maximum number of rounds / estimators for the training.
    :param early_stopping_rounds: If no improvement of the validation score in
    n rounds occur, the training will be stopped.
    :param output_filepath: Directory that will contain the trained models.
    """
    with timer("Building model and start training"):
        train_lgb_df = lgb.Dataset(data=train_df, label=label)
        valid_sets = [train_lgb_df]
        lgbm_model = lgb.train(params=params,
                               train_set=train_lgb_df,
                               num_boost_round=num_boost_round,
                               valid_sets=valid_sets,
                               verbose_eval=verbose_eval,
                               early_stopping_rounds=early_stopping_rounds)
    with timer("Saving trained model"):
        save_model(output_filepath, lgbm_model)


def start_full_by_meter_run(train_df, label, params, verbose_eval, num_boost_round,
                            early_stopping_rounds, output_filepath):
    """
    Divides the data into the four meter types and trains a model on each one.
    :param train_df: DataFrame which contains the training data.
    :param label: A vector which contains the labels of the training data.
    :param params: Dictionary with the model parameters
    :param verbose_eval: The interval where training information is printed
    to console.
    :param num_boost_round: Maximum number of rounds / estimators for the training.
    :param early_stopping_rounds: If no improvement of the validation score in
    n rounds occur, the training will be stopped.
    :param output_filepath: Directory that will contain the trained models.
    """
    output_filepath = output_filepath + "_by_meter"
    train_by_meter = []
    label_by_meter = []
    for i in range(4):
        is_meter = train_df["meter"] == i
        train_temp = train_df[is_meter]
        label_temp = label[is_meter]
        train_by_meter.append(train_temp)
        label_by_meter.append(label_temp)

    with timer("Building models and start training"):
        for (train, label) in zip(train_by_meter, label_by_meter):
            del train["meter"]
            train_lgb_df = lgb.Dataset(data=train, label=label)
            valid_sets = [train_lgb_df]
            lgbm_model = lgb.train(params=params,
                                   train_set=train_lgb_df,
                                   num_boost_round=num_boost_round,
                                   valid_sets=valid_sets,
                                   verbose_eval=verbose_eval,
                                   early_stopping_rounds=early_stopping_rounds)
            with timer("Saving trained model"):
                save_model(output_filepath, lgbm_model)


def start_full_by_building_run(train_df, label, params, splits, verbose_eval,
                               num_boost_round, early_stopping_rounds, output_filepath):
    """
    Trains a model for each of the buildings. Expect a high wall time as the
    count of the buildings is >1000.
    :param train_df: DataFrame which contains the training data.
    :param label: A vector which contains the labels of the training data.
    :param params: Dictionary with the model parameters
    :param splits: Integer describing the number of folds / splitting fraction.
    :param verbose_eval: The interval where training information is printed
    to console.
    :param num_boost_round: Maximum number of rounds / estimators for the training.
    :param early_stopping_rounds: If no improvement of the validation score in
    n rounds occur, the training will be stopped.
    :param output_filepath: Directory that will contain the trained models.
    """
    output_main_dir = output_filepath + "_by_building"
    train_df["label"] = label
    train_df = train_df.drop(columns=["site_id"], axis=1)
    train_df = train_df.groupby("building_id")
    buildings = [name for name, _ in train_df]

    for b in buildings:
        click.echo("Starting training for Building " + str(b) + ".")
        train_by_building = train_df.get_group(b)
        train_by_building = train_by_building.reset_index(drop=True)
        label = train_by_building["label"]
        train_by_building = train_by_building.drop(columns=["building_id", "label"], axis=1)
        with timer("Performing " + str(splits) + " fold cross-validation on building "\
                   + str(b)):
            kf = KFold(n_splits=splits, shuffle=False, random_state=1337)
            for i, (train_index, test_index) in enumerate(kf.split(train_by_building, label)):
                with timer("~~~~ Fold %d of %d ~~~~" % (i + 1, splits)):
                    x_train, x_valid = train_by_building.iloc[train_index], train_by_building.iloc[test_index]
                    y_train, y_valid = label[train_index], label[test_index]

                    train_lgb_df = lgb.Dataset(data=x_train, label=y_train)
                    valid_lgb_df = lgb.Dataset(data=x_valid, label=y_valid)

                    valid_sets = [train_lgb_df, valid_lgb_df]
                    evals_result = dict()
                    lgbm_model = lgb.train(params=params,
                                           train_set=train_lgb_df,
                                           num_boost_round=num_boost_round,
                                           valid_sets=valid_sets,
                                           valid_names=["train_loss", "eval"],
                                           verbose_eval=verbose_eval,
                                           evals_result=evals_result,
                                           early_stopping_rounds=early_stopping_rounds)
                    output_filepath = output_main_dir + "/" + str(b)
                    save_model(output_filepath, lgbm_model)


def start_cv_run(train_df, label, params, splits, verbose_eval,
                 num_boost_round, early_stopping_rounds, output_filepath,
                 grouped_on_building):
    """
    Starts a Cross Validation Run with the parameters provided.
    Scores will be documented and models will be saved.
    :param train_df: DataFrame which contains the training data.
    :param label: A vector which contains the labels of the training data.
    :param params: Dictionary with the model parameters
    :param splits: Integer describing the number of folds / splitting fraction.
    :param verbose_eval: The interval where training information is printed
    to console.
    :param num_boost_round: Maximum number of rounds / estimators for the training.
    :param early_stopping_rounds: If no improvement of the validation score in
    n rounds occur, the training will be stopped.
    :param output_filepath: Directory that will contain the trained models.
    :param grouped_ob_building: Logical indicating whether cross-validation should
    be done with Grouped-CV, only using readings of meter 0
    """
    if grouped_on_building:
        output_filepath = output_filepath + "_grouped_cv"
        train_df = train_df[train_df.meter == 0]
        label = train_df.meter_reading
        groups = train_df.building_id
        gkf = GroupKFold(n_splits = splits)
        indices = gkf.split(train_df, label, groups)
    else:
        output_filepath = output_filepath + "_cv"
        kf = Kfold(nsplits = splits, shuffle = False, random = state = 1337)
        indices = kf.split(train_df, label)
    cv_results = []
    with timer("Performing " + str(splits) + " fold cross-validation"):
        for i, (train_index, test_index) in enumerate(indices):
            with timer("~~~~ Fold %d of %d ~~~~" % (i + 1, splits)):
                x_train, x_valid = train_df.iloc[train_index], train_df.iloc[test_index]
                y_train, y_valid = label[train_index], label[test_index]

                train_lgb_df = lgb.Dataset(data=x_train, label=y_train)
                valid_lgb_df = lgb.Dataset(data=x_valid, label=y_valid)

                valid_sets = [train_lgb_df, valid_lgb_df]
                evals_result = dict()
                lgbm_model = lgb.train(params=params,
                                       train_set=train_lgb_df,
                                       num_boost_round=num_boost_round,
                                       valid_sets=valid_sets,
                                       valid_names=["train_loss", "eval"],
                                       verbose_eval=verbose_eval,
                                       evals_result=evals_result,
                                       early_stopping_rounds=early_stopping_rounds)
                save_model(output_filepath, lgbm_model)

                cv_results.append(evals_result)
        evaluate_cv_results(cv_results)


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
    model.save_model(output_filepath + "/" + new_version + ".txt", num_iteration=model.best_iteration)
    click.echo("Model successfully saved in folder: " + output_filepath)


def evaluate_cv_results(cv_results):
    """
    Prints overview of the respective folds and stores the result in
    models/cv_eval.
    :param cv_results: Dictionary with the logged training information.
    """
    summary = {
        "fold": [],
        "eval_loss": [],
        "train_loss": []
    }

    click.echo("Fold summary (Loss):")
    for i, metrics in enumerate(cv_results):
        eval_loss = float([x[-1] for x in list(metrics["eval"].values())][0])
        train_loss = float([x[-1] for x in list(metrics["train_loss"].values())][0])
        print("{0}|\tEval:\t{1:.3f}\t|\ttrain:\t{2:.3f}".format(i, eval_loss, train_loss))
        summary["fold"].append(i)
        summary["eval_loss"].append(eval_loss)
        summary["train_loss"].append(train_loss)

    avg_eval_loss = sum(summary["eval_loss"]) / len(summary["eval_loss"])
    avg_train_loss = sum(summary["train_loss"]) / len(summary["train_loss"])
    print("Average Eval Loss:\t{0:.3f}\nAverage Train Loss:\t{1:.3f}".format(avg_eval_loss, avg_train_loss))

    summary = pd.DataFrame.from_dict(summary)
    csv_path = "models/cv_eval"
    os.makedirs(csv_path, exist_ok=True)
    files_in_dir = os.listdir(csv_path)
    max_version = max([int(file[:4]) for file in files_in_dir], default=0)
    new_version = str(max_version + 1).zfill(4)
    summary.to_csv(csv_path + "/" + new_version + ".csv")


if __name__ == '__main__':
    main()
