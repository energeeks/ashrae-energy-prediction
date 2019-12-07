import os

import click
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as ctb
import yaml

from src.timer import timer


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_type')
@click.argument('model_path', type=click.Path(exists=True))
def main(input_filepath, model_type, model_path):
    """
    Loads a trained model and testing data to create a submission file which is
    ready for uploading.
    """
    with open("src/config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    with timer("Loading testing data"):
        test_df = pd.read_pickle(input_filepath + "/test_data.pkl")

    row_ids = test_df["row_id"]
    del test_df["row_id"]

    if model_type == "xgb":
        predictions = predict_with_xgb(test_df, model_path)

    elif model_type == "lgbm":
        predictions = predict_with_lgbm(test_df, row_ids, model_path)

    elif model_type == "ctb":
        predictions = predict_with_ctb(test_df, row_ids, model_path)

    elif model_type == "lgbm_meter":
        predictions = predict_with_lgbm_meter(test_df, row_ids, model_path)

    elif model_type == "lgbm_building":
        predictions = predict_with_lgbm_building(test_df, row_ids, model_path)

    else:
        raise ValueError(model_type + " is not a valid model type to predict from")

    with timer("Creating submission file"):
        create_submission_file(row_ids, predictions, cfg["use_leaks"])


def predict_with_xgb(test_df, model_filepath):
    """
    Loads the specified model and predicts the target variable which is being
    returned as list.
    """
    test_dmatrix = xgb.DMatrix(test_df)
    del test_df

    with timer("Loading model " + model_filepath):
        xgb_model = xgb.Booster()
        xgb_model.load_model(model_filepath)

    with timer("Predicting values"):
        predictions = xgb_model.predict(test_dmatrix)
        # Invert log and set possible neg. values to 0
        predictions = np.expm1(predictions)
        predictions[predictions < 0] = 0
    return predictions


def predict_with_lgbm(test_df, row_ids, model_filepath):
    """
    Loads the specified model and predicts the target variable which is being
    returned as list.
    """
    if os.path.isdir(model_filepath):
        click.echo("Loading models in directory" + model_filepath)
        models_in_dir = os.listdir(model_filepath)
        num_models = len(models_in_dir)
        predictions = np.zeros(len(row_ids))

        for i, model in enumerate(models_in_dir, start=1):
            with timer("Loading model [" + str(i) + "/" + str(num_models) + "]"):
                lgbm_model = lgb.Booster(model_file=model_filepath + "/" + model)

            with timer("Predicting values [" + str(i) + "/" + str(num_models) + "]"):
                predictions_current = lgbm_model.predict(test_df)
                predictions += np.expm1(predictions_current)

        predictions = predictions / num_models
        predictions[predictions < 0] = 0
        return predictions

    else:
        with timer("Loading model " + model_filepath):
            lgbm_model = lgb.Booster(model_file=model_filepath)

        with timer("Predicting values"):
            predictions = lgbm_model.predict(test_df)
            # Invert log and set possible neg. values to 0
            predictions = np.expm1(predictions)
        predictions[predictions < 0] = 0
        return predictions


def predict_with_ctb(test_df, row_ids, model_filepath):
    """
    Loads the specified model and predicts the target variable which is being
    returned as list.
    """
    if os.path.isdir(model_filepath):
        click.echo("Loading models in directory" + model_filepath)
        models_in_dir = os.listdir(model_filepath)
        num_models = len(models_in_dir)
        predictions = np.zeros(len(row_ids))

        for i, model in enumerate(models_in_dir, start=1):
            with timer("Loading model [" + str(i) + "/" + str(num_models) + "]"):
                ctb_model = ctb.CatBoostRegressor()
                ctb_model.load_model(model_filepath + "/" + model)

            with timer("Predicting values [" + str(i) + "/" + str(num_models) + "]"):
                predictions_current = ctb_model.predict(test_df)
                predictions += np.expm1(predictions_current)

        predictions = predictions / num_models
        predictions[predictions < 0] = 0
        return predictions

    else:
        with timer("Loading model " + model_filepath):
            ctb_model = ctb.CatBoostRegressor()
            ctb_model.load_model(model_filepath)

        with timer("Predicting values"):
            predictions = ctb_model.predict(test_df)
            # Invert log and set possible neg. values to 0
            predictions = np.expm1(predictions)
        predictions[predictions < 0] = 0
        return predictions


def predict_with_lgbm_meter(test_df, row_ids, model_filepath):
    """"
    Takes a given directory which contains four models (one for each
    meter type) and then predicts the rows with the respective model
    """

    meters_in_dir = sorted(os.listdir(model_filepath), key=int)
    test_df["row_id"] = row_ids
    test_df = test_df.groupby("meter")

    predictions_by_meter = []
    row_id_by_meter = []
    for m in meters_in_dir:
        test_by_meter = test_df.get_group(int(m))
        test_by_meter = test_by_meter.reset_index(drop=True)
        rows_grouped = list(test_by_meter["row_id"])
        test_by_meter = test_by_meter.drop(columns=["meter", "row_id"], axis=1)

        models_in_dir = os.listdir(model_filepath + "/" + m)
        num_models = len(models_in_dir)
        predictions_group = np.zeros(len(rows_grouped))
        i = 0
        for model in models_in_dir:
            i += 1
            click.echo("Predicting meter " + m + " [" + str(i) + "/" + str(num_models) + "]")
            lgbm_model = lgb.Booster(model_file=model_filepath + "/" + m + "/" + model)
            predictions_current = lgbm_model.predict(test_by_meter)
            predictions_group += np.expm1(predictions_current)

        predictions_group = predictions_group / num_models
        predictions_by_meter.extend(list(predictions_group))
        row_id_by_meter.extend(rows_grouped)

    # Order the predictions by merging them to the original row ids
    pred_df = pd.DataFrame({"row_id": row_id_by_meter, "pred": predictions_by_meter})
    pred_df = pred_df.sort_values("row_id")
    predictions = pred_df["pred"].copy(deep=True)
    predictions[predictions < 0] = 0
    return predictions


def predict_with_lgbm_building(test_df, row_ids, model_filepath):
    """"
    Takes a given directory which contains n folders (one for each
    building) and then predicts the rows with the respective models
    """
    buildings_in_dir = sorted(os.listdir(model_filepath), key=int)
    test_df["row_id"] = row_ids
    test_df = test_df.drop(columns=["site_id"], axis=1)
    test_df = test_df.groupby("building_id")

    predictions_by_building = []
    row_id_by_building = []
    for b in buildings_in_dir:
        test_by_building = test_df.get_group(int(b))
        test_by_building = test_by_building.reset_index(drop=True)
        rows_grouped = list(test_by_building["row_id"])
        test_by_building = test_by_building.drop(columns=["building_id"], axis=1)

        models_in_dir = os.listdir(model_filepath + "/" + b)
        num_models = len(models_in_dir)
        predictions_group = np.zeros(len(rows_grouped))
        i = 0
        for model in models_in_dir:
            i += 1
            click.echo("Predicting Building " + b + " [" + str(i) + "/" + str(num_models) + "]")
            lgbm_model = lgb.Booster(model_file=model_filepath + "/" + b + "/" + model)
            predictions_current = lgbm_model.predict(test_by_building)
            predictions_group += np.expm1(predictions_current)

        predictions_group = predictions_group / num_models
        predictions_by_building.extend(list(predictions_group))
        row_id_by_building.extend(rows_grouped)

    # Order the predictions by merging them to the original row ids
    pred_df = pd.DataFrame({"row_id": row_id_by_building, "pred": predictions_by_building})
    pred_df = pred_df.sort_values("row_id")
    predictions = pred_df["pred"].copy(deep=True)
    predictions[predictions < 0] = 0
    return predictions


def create_submission_file(row_ids, predictions, use_leaks=False):
    """
    Creates a submission file which fulfills the upload conditions for the
    kaggle challenge.
    """
    if use_leaks:
        with timer("Adding leaks to submission file"):
            predictions = add_leaks_to_submission(predictions)

    submission = pd.DataFrame({"row_id": row_ids, "meter_reading": predictions})

    validate_submission(submission)

    submission_dir = "submissions"
    os.makedirs(submission_dir, exist_ok=True)
    submission.to_csv(submission_dir + "/submission.csv", index=False)


def validate_submission(submission_df):
    submission_error = get_submission_error(submission_df)
    if submission_error:
        click.secho(submission_error, err=True, fg="red", bold=True)
        click.pause()


def get_submission_error(submission_df):
    actual_columns = submission_df.columns
    expected_columns = ["row_id", "meter_reading"]
    if list(actual_columns) != expected_columns:
        return "Submission has incorrect columns: " + str(list(actual_columns)) + ", expected: " + str(expected_columns)

    actual_row_count = len(submission_df)
    expected_row_count = 41697600
    if actual_row_count < expected_row_count:
        return "Submission has to few rows: " + str(actual_row_count) + ", expected: " + str(expected_row_count)
    if actual_row_count > expected_row_count:
        return "Submission has to many rows: " + str(actual_row_count) + ", expected: " + str(expected_row_count)

    if any(submission_df["row_id"].values != np.arange(expected_row_count)):
        return "Submission has incorrect row_ids"

    return None


def add_leaks_to_submission(predictions):
    """"
    Complements the predicted values with the real leaked labels. Special thanks to
    https://www.kaggle.com/yamsam/ashrae-leak-data-station
    """
    leaked_df = pd.read_feather("data/leak/leak.feather")
    leaked_df.rename(columns={"meter_reading": "leaked_reading"}, inplace=True)
    leaked_df.loc[leaked_df["leaked_reading"] < 0, "leaked_reading"] = 0
    leaked_df = leaked_df[leaked_df["building_id"] != 245]
    leaked_df["timestamp"] = leaked_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    test_df = pd.read_csv("data/raw/test.csv")

    test_df = test_df.merge(leaked_df, left_on=["building_id", "meter", "timestamp"],
                            right_on=["building_id", "meter", "timestamp"], how="left")
    test_df["meter_reading"] = predictions
    test_df["meter_reading"] = np.where(test_df["leaked_reading"].isna(),
                                        test_df["meter_reading"], test_df["leaked_reading"])

    return test_df["meter_reading"]


if __name__ == '__main__':
    main()
