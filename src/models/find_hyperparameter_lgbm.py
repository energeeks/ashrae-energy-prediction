# This is merely a script which kicks of a hyperparameter search using
# Bayesian optimization.
# Thanks to https://github.com/MBKraus/Hyperopt for input and ideas!
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from src.timer import timer


def main():
    ################################################################################
    # SET PARAMETER FOR SEARCH HERE
    ################################################################################
    params_hyperopt = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
        'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
        'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
        'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    }

    splits = 5

    ################################################################################
    with timer("Loading processed training data"):
        train_df, label = load_processed_training_data("data/processed")
        train_df = lgb.Dataset(train_df, label)

    def objective_function(params):
        kf = KFold(n_splits=splits, shuffle=False, random_state=1337)
        cv_results = lgb.cv(params,
                            train_df,
                            folds=kf,
                            num_boost_round=10000,
                            early_stopping_rounds=100,
                            metrics="rmse",
                            seed=1337)

        score = min(cv_results["rmse-mean"])
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function,
                      params_hyperopt,
                      algo=tpe.suggest,
                      max_evals=100,
                      trials=trials,
                      rstate=1337)


def load_processed_training_data(input_filepath):
    """
    Loads processed data and returns a xgb Matrix with distinguished label
    column.
    """
    train_df = pd.read_pickle(input_filepath + "/train_data.pkl")

    label = np.log1p(train_df["meter_reading"])
    del train_df["meter_reading"]

    return train_df, label


if __name__ == '__main__':
    main()
