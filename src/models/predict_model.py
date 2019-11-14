import os
import click
import pandas as pd
import xgboost as xgb


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_type')
@click.argument('model_path', type=click.Path(exists=True))
def main(input_filepath, model_type, model_path):
    """
    Loads a trained model and testing data to create a submission file which is
    ready for uploading.
    """
    click.echo("Loading testing data...")
    test_df = pd.read_pickle(input_filepath + "/test_data.pkl")

    row_ids = test_df["row_id"]
    del test_df["row_id"]

    if model_type == "xgb":
        predictions = predict_with_xgb(test_df, model_path)
    else:
        raise ValueError(model_type + " is not a valid model type to predict from")

    click.echo("Creating submission file...")
    create_submission_file(row_ids, predictions)


def predict_with_xgb(test_df, model_filepath):
    """
    Loads the specified model and predicts the target variable which is being
    returned as list.
    """
    test_dmatrix = xgb.DMatrix(test_df)
    del test_df

    click.echo("Loading model " + model_filepath + "...")
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_filepath)

    click.echo("Predicting values...")
    predictions = xgb_model.predict(test_dmatrix)
    return predictions


def create_submission_file(row_ids, predictions):
    """
    Creates a submission file which fulfills the upload conditions for the
    kaggle challenge.
    """
    submission = pd.DataFrame({"row_id": row_ids, "meter_reading": predictions})
    submission_dir = "submissions"
    os.makedirs(submission_dir, exist_ok=True)
    submission.to_csv(submission_dir + "/submission.csv", index=False)


if __name__ == '__main__':
    main()
