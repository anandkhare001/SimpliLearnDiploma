import config
from data import load_pipeline
import NLP
from sklearn.feature_extraction.text import TfidfVectorizer


def predict_review(review):
    review = NLP.normalise_text(review)
    review = NLP.tokenize_text(review)
    review = NLP.del_stop(review)
    review = NLP.clean_text(review)
    Tf = TfidfVectorizer(max_features=10)
    review = Tf.fit_transform(review)

    model = load_pipeline(config.MODEL_NAME)
    rating = model.best_estimator_.predict(review)
    print(rating)
    return rating


if __name__ == '__main__':
    reviews = ["Really lovely place for steaks and sizzlers. It was so flavourful and delicious. "
               "The herb sauce is one of its kind. Must try!!"]
    predict_review(reviews)


