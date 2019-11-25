import os
import yaml
import click
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta
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

    elif model_type == "lgbm_meter":
        predictions = predict_with_lgbm_meter(test_df, row_ids, model_path)

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


def predict_with_lgbm_meter(test_df, row_ids, model_filepath):
    """"
    Takes a given directory which contains four models (one for each
    meter type) and then predicts the rows with the respective model
    """

    with timer("Loading models in directory" + model_filepath):
        models_in_dir = sorted(os.listdir(model_filepath))
        test_by_meter = []
        row_id_by_meter = []
        for i in range(4):
            is_meter = test_df["meter"] == i
            test_temp = test_df[is_meter]
            row_temp = row_ids[is_meter]
            test_by_meter.append(test_temp)
            row_id_by_meter.append(row_temp)

    predictions = []
    row_ids_prediction = []
    with timer("Predicting values"):
        for model, test, row in zip(models_in_dir, test_by_meter, row_id_by_meter):
            del test["meter"]
            lgbm_model = lgb.Booster(model_file=model_filepath + "/" + model)

            predictions_current = lgbm_model.predict(test)
            predictions.extend(list(np.expm1(predictions_current)))
            row_ids_prediction.extend(row)

    # Order the predictions by merging them to the original row ids
    pred_df = pd.DataFrame({"row_id": row_ids_prediction, "pred": predictions})
    row_ids_df = pd.DataFrame({"true_row_ids": row_ids})
    pred_ordered_df = row_ids_df.merge(pred_df, left_on="true_row_ids",
                                       right_on="row_id", how="left")
    predictions = pred_ordered_df["pred"].copy(deep=True)
    predictions[predictions < 0] = 0
    return predictions


def create_submission_file(row_ids, predictions, use_leaks=False):
    """
    Creates a submission file which fulfills the upload conditions for the
    kaggle challenge.
    """
    submission = pd.DataFrame({"row_id": row_ids, "meter_reading": predictions})

    if use_leaks:
        submission = add_leaks_to_submission(submission)

    submission_dir = "submissions"
    os.makedirs(submission_dir, exist_ok=True)
    submission.to_csv(submission_dir + "/submission.csv", index=False)


def add_leaks_to_submission(submission):
    """"
    Complements the predicted values with the real leaked labels. Special thanks to
    https://www.kaggle.com/wuliaokaola/ashrae-maybe-this-can-make-public-lb-some-useful
    """
    test_df = pd.read_csv("data/raw/test.csv", index_col=0, parse_dates=['timestamp'])
    building_df = pd.read_csv("data/raw/building_metadata.csv", usecols=['site_id', 'building_id'])
    test_df = test_df.merge(building_df, left_on="building_id", right_on="building_id", how="left")
    t = test_df[['building_id', 'meter', 'timestamp']]
    t['row_id'] = t.index

    # Site 0 Leaks
    site_0 = pd.read_csv("data/leaks/submission_ucf_replaced.csv", index_col=0)
    submission.loc[test_df[test_df["site_id"] == 0].index, "meter_reading"] = site_0["meter_reading"]

    # Site 1 Leaks
    site_1 = pd.read_pickle("data/leaks/site1.pkl")
    site_1 = site_1[site_1['timestamp'].dt.year > 2016]
    site_1 = site_1.merge(t, left_on=["building_id", "meter", "timestamp"],
                          right_on=["building_id", "meter", "timestamp"], how="left")
    site_1 = site_1[["meter_reading_scraped", "row_id"]].set_index("row_id").dropna()
    submission.loc[site_1.index, "meter_reading"] = site_1["meter_reading_scraped"]

    # Site 2 Leaks
    site_2 = pd.read_csv("data/leaks/asu_2016-2018.csv", parse_dates=["timestamp"])
    site_2 = site_2[site_2["timestamp"].dt.year > 2016]
    site_2 = site_2.merge(t, left_on=["building_id", "meter", "timestamp"],
                          right_on=["building_id", "meter", "timestamp"], how="left")
    site_2 = site_2[["meter_reading", "row_id"]].set_index("row_id").dropna()
    submission.loc[site_2.index, "meter_reading"] = site_2["meter_reading"]

    # Site 4 Leaks
    site_4 = pd.read_csv("data/leaks/site4_leak_data_v2.csv", parse_dates=["timestamp"])
    site_4["meter"] = 0
    site_4["timestamp"] = pd.DatetimeIndex(site_4["timestamp"]) + timedelta(hours=-8)
    site_4 = site_4[site_4["timestamp"].dt.year > 2016]
    site_4 = site_4.merge(t, left_on=["building_id", "meter", "timestamp"],
                          right_on=["building_id", "meter", "timestamp"], how="left")
    site_4 = site_4[["meter_reading", "row_id"]].set_index("row_id").dropna()
    submission.loc[site_4.index, "meter_reading"] = site_4["meter_reading"]

    return submission


if __name__ == '__main__':
    main()
