import streamlit as st
import joblib as jb
import numpy as np
import NLP
from sklearn.feature_extraction.text import TfidfVectorizer

acknowledge = ""


def predict(review):
    review = NLP.normalise_text(review)
    review = NLP.tokenize_text(review)
    review = NLP.del_stop(review)
    review = NLP.clean_text(review)
    Tf = TfidfVectorizer(max_features=10)
    review = Tf.fit_transform(review)

    model = jb.load("ZomatoReviewModel.pkl")
    rating = model.best_estimator_.predict(review)
    return rating


def main():
    global acknowledge
    st.title("Welcome to  Zomato Reviews Section")
    st.header("Kindly submit your valuable feedback")
    review = st.text_input("Share your feedback")
    submit = st.button("Submit Review")

    if submit:
        rating = predict(review)

        if 3 < rating < 4.5:
            acknowledge = f"""
                              Thank you for your valuable feedback.
                              Your positive feedback motivates us to continuously improve our services.
                              """
        elif rating > 4.5:
            acknowledge = f"""
                              Thank you for your valuable feedback.
                              We are glad that you liked our services.
                              We'll strive to maintain the same quality of our services.
                              """

        elif rating < 3:
            acknowledge = f"""
                              Thank you for your valuable feedback.
                              We're sorry that you had inconvenience with our services.
                              We'll take this feedback as an opportunity for improvement and strive  to better.
                              """
        else:
            acknowledge = ""

        st.success(acknowledge)


if __name__ == '__main__':
    main()
