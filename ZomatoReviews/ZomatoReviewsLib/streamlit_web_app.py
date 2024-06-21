import streamlit as st
import joblib as jb
import numpy as np
import config
from data import load_pipeline
import predict


model = load_pipeline(config.MODEL_NAME)


def main():
    st.title("Welcome to  Zomato Reviews Section")
    st.header("Kindly submit your valuable feedback")
    review = st.text_input("Share your feedback", height=300)
    submit = st.button("Submit Review")

    if submit:
        rating = predict.predict_review(review)

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
        st.write(acknowledge, height=100)


if __name__ == "__main__":
    main()
