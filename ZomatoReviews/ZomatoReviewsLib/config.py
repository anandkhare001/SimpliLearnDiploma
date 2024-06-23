import pathlib
import os
import ZomatoReviewsLib

PACKAGE_ROOT = pathlib.Path(ZomatoReviewsLib.__file__).resolve().parent
DATA_PATH = os.path.join(PACKAGE_ROOT, "datasets")
TRAIN_FILE = 'Zomato_reviews.csv'
MODEL_NAME = 'ZomatoReviewModel.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")
TARGET = 'rating'
FEATURES = 'review_text'
MAX_DEPTH = [10, 15, 20, 25]
