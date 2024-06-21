import mlflow
import argparse
from train import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_function(actual, pred):
    rmse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main(criterion, max_depth, n_estimators, max_features):
    # Preprocess and extract input and output features
    xx, yy = preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.3, random_state=42)

    mlflow.set_experiment('ZomatoReviews')
    with mlflow.start_run():
        mlflow.set_tracking_uri("http://127.0.0.1:2000")

        # Log Input params
        mlflow.set_tag('version', '1.0.0')
        mlflow.log_param('max_features', max_features)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('criterion', criterion)
        mlflow.log_param('n_estimators', n_estimators)

        # Vectorise the reviews
        vectorizer = TfidfVectorizer(max_features=10)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.fit_transform(X_test)

        # Model Training
        model = RandomForestRegressor(criterion=criterion,
                                      max_depth=max_depth,
                                      n_estimators=n_estimators,
                                      random_state=101)
        model.fit(X_train_bow, y_train)

        # Metric calculation
        y_pred = model.predict(X_test_bow)
        rmse, mae, r2 = eval_function(y_test, y_pred)

        # Log Metrics
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2_score', r2)
        mlflow.sklearn.log_model(model, 'trained_model')  # Model var, Folder name


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--criterion', type=str, default='square_error')
    args.add_argument('--max_depth', type=int, default=None)
    args.add_argument('--n_estimators', type=int, default=100)
    args.add_argument('--max_features', type=int, default=10)
    parsed_args = args.parse_args()

    main(parsed_args.criterion, parsed_args.max_depth, parsed_args.n_estimators, parsed_args.max_features)
