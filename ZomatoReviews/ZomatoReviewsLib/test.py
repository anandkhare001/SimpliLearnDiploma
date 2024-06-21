from train import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from data import *
import NLP
import config
import re


#model = load_pipeline(config.MODEL_NAME)
#print(model)
#print(model.best_estimator_)

#review = 'My Name is Anand. I am a Data Scientist. I develop good ML models.'
review = ["Really lovely place for steaks and sizzlers. It was so flavourful and delicious. The herb sauce is one of its kind. Must try!!"]
review = NLP.normalise_text(review)
review = NLP.tokenize_text(review)
review = NLP.del_stop(review)
review = NLP.clean_text(review)
print(review)
