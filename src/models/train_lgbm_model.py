import os
import click
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold


@click.command()
@click.argument('mode')
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(mode, input_filepath, output_filepath):
    """
    Collects prepared data and starts training an LightGBM model. Parameters
    can be specified by editing the main function of .py file
    """
    click.echo("Loading processed training data...")
    train_df, label = load_processed_training_data(input_filepath)

    ###########################################################################
    # DEFINE PARAMETERS FOR THE LGBM MODEL                                     #
    ###########################################################################
    categorical_features = ["building_id", "site_id", "meter", "primary_use"]

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 40,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse"
    }

    num_boost_round = 5
    early_stopping_rounds = 2

    ###########################################################################

    if mode == "full":
        start_full_training_run(train_df, label, categorical_features, params,
                                num_boost_round, early_stopping_rounds,
                                output_filepath)

    elif mode == "cv":
        start_cv_run(train_df, label, categorical_features, params,
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


def start_full_training_run(train_df, label, categorical_features, params,
                            num_boost_round, early_stopping_rounds, output_filepath):
    """"
    Starts a full training run with the provided parameters.
    """
    click.echo("Building model and start training...")
    train_lgb_df = lgb.Dataset(data=train_df, label=label,
                               categorical_feature=categorical_features)
    verbose_eval = 25
    valid_sets = [train_lgb_df]
    lgbm_model = lgb.train(params=params,
                           train_set=train_lgb_df,
                           num_boost_round=num_boost_round,
                           valid_sets=valid_sets,
                           verbose_eval=verbose_eval,
                           early_stopping_rounds=early_stopping_rounds)
    click.echo("Saving trained model...")
    save_model(output_filepath, lgbm_model)


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


def start_cv_run(train_df, label, categorical_features, params,
                 num_boost_round, early_stopping_rounds, output_filepath):
    cv_results = []
    splits = 2
    click.echo("Starting " + str(splits) + " fold cross-validation...")
    kf = KFold(n_splits=splits, shuffle=False, random_state=1337)
    for i, (train_index, test_index) in enumerate(kf.split(train_df, label)):
        click.echo(("~~~~ Fold %d of %d ~~~~" % (i + 1, splits)))
        x_train, x_valid = train_df.iloc[train_index], train_df.iloc[test_index]
        y_train, y_valid = label[train_index], label[test_index]

        train_lgb_df = lgb.Dataset(data=x_train, label=y_train,
                                   categorical_feature=categorical_features)
        valid_lgb_df = lgb.Dataset(data=x_valid, label=y_valid,
                                   categorical_feature=categorical_features)

        valid_sets = [train_lgb_df, valid_lgb_df]
        verbose_eval = True
        evals_result = dict()
        lgbm_model = lgb.train(params=params,
                               train_set=train_lgb_df,
                               num_boost_round=num_boost_round,
                               valid_sets=valid_sets,
                               verbose_eval=verbose_eval,
                               evals_result=evals_result,
                               early_stopping_rounds=early_stopping_rounds)
        output_filepath = output_filepath + "_cv"
        save_model(output_filepath, lgbm_model)

        cv_results.append(evals_result)
    evaluate_xgb_cv_results(cv_results)


def evaluate_xgb_cv_results(cv_results):
    """
    Prints overview of the respective folds and stores the result in
    models/xgb_cv.
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
    csv_path = "models/xgb_cv"
    os.makedirs(csv_path, exist_ok=True)
    files_in_dir = os.listdir(csv_path)
    max_version = max([int(file[:4]) for file in files_in_dir], default=0)
    new_version = str(max_version + 1).zfill(4)
    summary.to_csv(csv_path + "/" + new_version + ".csv")


if __name__ == '__main__':
    main()
