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
    can be specified by editing the main function of .py file
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
    num_boost_round = cfg["num_boost_round"]
    early_stopping_rounds = cfg["early_stopping_rounds"]
    splits = cfg["splits_for_cv"]
    verbose_eval = cfg["verbose_eval"]
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
    Loads processed data and returns a xgb Matrix with distinguished label
    column.
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")

    label = np.log1p(train_df["meter_reading"])
    del train_df["meter_reading"]

    return train_df, label


def start_full_training_run(train_df, label, params, verbose_eval,
                            num_boost_round, early_stopping_rounds, output_filepath):
    """"
    Starts a full training run with the provided parameters.
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
    Divides the data into the four meter types and trains a model on each one
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
    """
    output_main_dir = output_filepath + "_by_building"
    train_df["label"] = label
    train_df = train_df.drop(columns=["site_id"], axis=1)

    for train in train_df.groupby("building_id"):
        building = train_df["building_id"].loc[0]
        click.echo("Starting training for Building " + str(building) + ".")
        train_by_building = train.reset_index(drop=True)
        label = train_by_building["label"]
        train_by_building = train_by_building.drop(columns=["building_id", "label"], axis=1)
        with timer("Performing " + str(splits) + " fold cross-validation on \
        building " + str(building)):
            kf = KFold(n_splits=splits, shuffle=False, random_state=1337)
            for i, (train_index, test_index) in enumerate(kf.split(train_by_building, label)):
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
                    output_filepath = output_main_dir + "/" + str(building)
                    save_model(output_filepath, lgbm_model)


def start_cv_run(train_df, label, params, splits, verbose_eval,
                 num_boost_round, early_stopping_rounds, output_filepath):
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
    Saves the trained model in /models/lgbm
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
