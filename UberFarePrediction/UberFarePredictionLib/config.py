import pathlib
import os
import UberFarePredictionLib

PACKAGE_ROOT = pathlib.Path(UberFarePredictionLib.__file__).resolve().parent
DATA_PATH = os.path.join(PACKAGE_ROOT, "datasets")
TRAIN_FILE = 'UberDataTrain.csv'
TEST_FILE = 'UberDataTest.csv'
FILTERED_TRAIN_DATA_FILE = 'Filtered_train_data.csv'
FILTERED_TEST_DATA_FILE = 'Filtered_test_data.csv'
MODEL_NAME = 'UberFareModel.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")
TARGET = 'fare_amount'
FEATURES = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'passenger_count']  # Final features used in model
NUM_FEATURES = ['pickup_longitude', 'pickup_latitude',
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
                'day', 'hour', 'month', 'year']
CAT_FEATURES = ['weekday']
FEATURES_TO_ENCODE = 'pickup_datetime'
FEATURES_TO_MAKE_BOUNDARY = ['pickup_latitude', 'pickup_longitude',
                             'dropoff_latitude', 'dropoff_longitude']
DROP_FEATURES = ['key']

