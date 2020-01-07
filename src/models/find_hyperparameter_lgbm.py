# This is merely a script which kicks of a hyperparameter search using
# Bayesian optimization.
# Thanks to https://github.com/MBKraus/Hyperopt for input and ideas!
import os
import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from src.timer import timer


def main():
    """
    A hyperparameter search using the hyperopt package is being conducted.
    Parameters can be defined in the respective script. The results will be saved in
    data/hyperopt.
    """
    ################################################################################
    # SET PARAMETER FOR SEARCH HERE
    ################################################################################
    params_hyperopt = {
        "num_leaves": scope.int(hp.quniform("num_leaves", 5, 4096, 10)),
        "min_data_in_leaf": scope.int(hp.quniform("min_data_in_leaf", 10, 50, 1)),
        "feature_fraction": hp.uniform("feature_fraction", 0.4, 1.0),
        "min_split_gain": hp.uniform("min_split_gain", 0, 0.5),
        "bagging_fraction": hp.uniform("bagging_fraction", 0.4, 1.0),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 4.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 4.0),
    }

    params_static = {
        "learning_rate": 0.05,
        "num_threads": 20,
        "device_type": "cpu",
        "verbosity": -1
    }

    splits = 5
    max_evals = 200

    ################################################################################
    with timer("Loading processed training data"):
        train_df, label = load_processed_training_data("data/processed")
        train_df = lgb.Dataset(train_df, label)

    def objective_function(params):
        kf = KFold(n_splits=splits, shuffle=False, random_state=1337)
        final_params = dict(params, **params_static)
        cv_results = lgb.cv(final_params,
                            train_df,
                            folds=kf,
                            num_boost_round=10000,
                            early_stopping_rounds=50,
                            metrics="rmse",
                            seed=1337)

        score = min(cv_results["rmse-mean"])
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function,
                      params_hyperopt,
                      algo=tpe.suggest,
                      max_evals=max_evals,
                      trials=trials,
                      rstate=np.random.RandomState(1337))

    print("The search proposes these hyperparameters:")
    print(best_param)

    os.makedirs("data/hyperopt/lgbm", exist_ok=True)
    with open("data/hyperopt/lgbm/best_param.pkl", "wb") as handle:
        pickle.dump(best_param, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/hyperopt/lgbm/trials.pkl", "wb") as handle:
        pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_processed_training_data(input_filepath):
    """
    Loads processed data and returns a dataframe with distinguished label
    column.

    :param input_filepath: Directory that contains the processed data
    :return Tuple consisting of the loaded train dataframe and corresponding
    labels
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")

    label = np.log1p(train_df["meter_reading"])
    del train_df["meter_reading"]

    return train_df, label


if __name__ == '__main__':
    main()
