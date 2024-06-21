import os
import joblib
import numpy as np
import pandas as pd
import config


def load_data(file_name):
    file_path = os.path.join(config.DATA_PATH, file_name)
    data = pd.read_csv(file_path)
    return data


def save_pipeline(pipeline):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")


def load_pipeline(pipeline):
    load_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(load_path)
    return model_loaded


if __name__ == '__main__':
    train_data = load_data('Zomato_reviews.csv')
    print(train_data)
    print(type(train_data))
