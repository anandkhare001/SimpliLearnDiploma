import pandas as pd
import numpy as np
import joblib
import config
from data import load_pipeline, load_data
from sklearn.metrics import mean_squared_error, r2_score


def generate_predictions():
    test_data = load_data(config.FILTERED_TEST_DATA_FILE)
    rf_model = load_pipeline(config.MODEL_NAME)
    predictions = rf_model.predict(test_data)
    results = {'Predictions': predictions}
    return results


def get_prediction(data_input):
    data = pd.DataFrame(data_input)
    rf_model = load_pipeline(config.MODEL_NAME)
    prediction = rf_model.predict(data[config.FEATURES])
    results = {'predictions': prediction}
    return results


if __name__ == '__main__':
    res = generate_predictions()
    print(res)
