import numpy as np
import pandas as pd
import calendar
from UberFarePredictionLib import config
from UberFarePredictionLib import data


def encode_date_time(df):
    df['day'] = df[config.FEATURES_TO_ENCODE].apply(lambda x: x.day)
    df['hour'] = df[config.FEATURES_TO_ENCODE].apply(lambda x: x.hour)
    df['weekday'] = df[config.FEATURES_TO_ENCODE].apply(lambda x: calendar.day_name[x.weekday()])
    df['month'] = df[config.FEATURES_TO_ENCODE].apply(lambda x: x.month)
    df['year'] = df[config.FEATURES_TO_ENCODE].apply(lambda x: x.year)
    return df


def get_min_max_coordinates(df):
    min_latitude = min([df[config.FEATURES_TO_MAKE_BOUNDARY[0]].min(),
                        df[config.FEATURES_TO_MAKE_BOUNDARY[2]].min()])
    max_latitude = min([df[config.FEATURES_TO_MAKE_BOUNDARY[0]].max(),
                        df[config.FEATURES_TO_MAKE_BOUNDARY[2]].max()])
    min_longitude = min([df[config.FEATURES_TO_MAKE_BOUNDARY[1]].min(),
                         df[config.FEATURES_TO_MAKE_BOUNDARY[3]].min()])
    max_longitude = min([df[config.FEATURES_TO_MAKE_BOUNDARY[1]].max(),
                         df[config.FEATURES_TO_MAKE_BOUNDARY[3]].max()])
    return min_latitude, max_latitude, min_longitude, max_longitude


def preprocess(file_name):
    # Load train and test data
    df = data.load_data(file_name)

    # convert pickup date column to datatime format
    df[config.FEATURES_TO_ENCODE] = pd.to_datetime(df[config.FEATURES_TO_ENCODE])

    # split pickup date column into day, month, year columns to make them as numerical data
    df = encode_date_time(df)

    # Remove NaN entries is any
    df.dropna(inplace=True)

    # Get min and max coordinates to make the data boundary
    min_latitude_test, max_latitude_test, min_longitude_test, max_longitude_test = get_min_max_coordinates(df)

    # Remove all the entries outside this coordinates boundary
    tmp_df = df[(df[config.FEATURES_TO_MAKE_BOUNDARY[0]] <= min_latitude_test) |
                (df[config.FEATURES_TO_MAKE_BOUNDARY[2]] <= min_latitude_test) |
                (df[config.FEATURES_TO_MAKE_BOUNDARY[1]] <= min_longitude_test) |
                (df[config.FEATURES_TO_MAKE_BOUNDARY[3]] <= min_longitude_test) |
                (df[config.FEATURES_TO_MAKE_BOUNDARY[0]] >= max_latitude_test) |
                (df[config.FEATURES_TO_MAKE_BOUNDARY[2]] >= max_latitude_test) |
                (df[config.FEATURES_TO_MAKE_BOUNDARY[1]] >= max_longitude_test) |
                (df[config.FEATURES_TO_MAKE_BOUNDARY[3]] >= max_longitude_test)]
    df.drop(tmp_df.index, inplace=True)

    # Remove all rows where fare amount is negative
    if config.TARGET in df.columns:
        df = df[df[config.TARGET] > 0]

    # Encode weekday details to numerical
    df.weekday = df.weekday.map(
        {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6})

    # Drop pickup date as we created new features from it
    df.drop(['key', 'pickup_datetime'], axis=1, inplace=True)

    return df


if __name__ == '__main__':
    data_df = preprocess(config.TEST_FILE)
    #data_df = preprocess(config.TRAIN_FILE)
    print(data_df.head())
    print(type(data_df))

