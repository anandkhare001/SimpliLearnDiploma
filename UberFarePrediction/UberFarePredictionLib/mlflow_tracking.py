import os
import mlflow
import argparse
import time
import numpy as np
import pandas as pd

from UberFarePredictionLib import config
from UberFarePredictionLib import data
from UberFarePredictionLib import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(file_name):
    file_path = os.path.join(config.DATA_PATH, file_name)
    df = pd.read_csv(file_path)
    return df


def eval_function(actual, pred):
    rmse = mean_absolute_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main(criterion, max_depth, n_estimators):
    train = load_data('Filtered_train_data.csv')
    X = train.drop([config.TARGET], axis=1)
    y = train[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6, test_size=0.2)

    mlflow.set_experiment('UberFareModel')
    with mlflow.start_run():
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        run_id = run.info.run_id
        mlflow.set_tag('version', '1.0.0')
        mlflow.log_param('criterion', criterion)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('n_estimators', n_estimators)

        model = RandomForestRegressor(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators, random_state=101)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_function(y_test, y_pred)

        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)
        mlflow.sklearn.log_model(model, 'trained_model')  # model, folder name


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--criterion', type=str, default='squared_error')
    args.add_argument('--max_depth', type=int, default=None)
    args.add_argument('--n_estimators', type=int, default=100)
    parsed_args = args.parse_args()
    # parsed_args.param1
    main(parsed_args.criterion, parsed_args.max_depth, parsed_args.n_estimators)
