import os
import click
import pandas as pd
import xgboost as xgb


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Collects prepared data and starts training an xgb model. Parameters
    can be specified by editing the main function of .py file
    """
    click.echo("Loading processed training data...")
    train_dmatrix = load_processed_training_data(input_filepath)

    ###########################################################################
    # DEFINE PARAMETERS FOR THE XGB MODEL                                     #
    ###########################################################################

    params = {
        "objective": "reg:squaredlogerror",
        "eval_metric": "rmsle",
        "booster": "gbtree",
        "verbosity": "1",
    }

    num_boost_round = 10
    early_stopping_rounds = 10
    evals = [(train_dmatrix, 'eval')]
    verbose_eval = True

    ###########################################################################

    click.echo("Building model and start training...")
    xgb_model = xgb.train(params=params,
                          dtrain=train_dmatrix,
                          num_boost_round=num_boost_round,
                          evals=evals,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stopping_rounds)

    click.echo("Saving trained model...")
    save_model(output_filepath, xgb_model)


def load_processed_training_data(input_filepath):
    """
    Loads processed data and returns a xgb Matrix with distinguished label
    column.
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")

    ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!####
    ### IMPORTANT ######################
    # I'm taking only half of the training values for testing and until the
    # RAM issues are gone
    ####################################
    train_df = train_df.sample(n=len(train_df)//2)

    y_train = train_df["meter_reading"]
    del train_df["meter_reading"]

    return xgb.DMatrix(data=train_df, label=y_train)


def save_model(output_filepath, model):
    """
    Saves the trained model in /models/xgb
    """
    os.makedirs(output_filepath, exist_ok=True)
    files_in_dir = os.listdir(output_filepath)
    max_version = max([int(file[:-6]) for file in files_in_dir], default=0)
    new_version = str(max_version + 1).zfill(4)
    model.save_model(output_filepath + "/" + new_version + ".model")
    click.echo("Model successfully saved in folder: " + output_filepath)


if __name__ == '__main__':
    main()
