import pandas as pd
import numpy as np
import os
import re
import config


def remove_null_entries(text):
    text = text[~text[config.FEATURES].isnull()]
    text.reset_index(inplace=True, drop=True)
    return text


def text_to_list(text):
    text = text[config.FEATURES].values
    return text


def normalise_text(text):
    text = [d.lower() for d in text]
    text = [' '.join(txt.split()) for txt in text]
    return text


def tokenize_text(text):
    from nltk.tokenize import word_tokenize
    text = [word_tokenize(tkn) for tkn in text]
    return text


def del_stop(text):
    from nltk.corpus import stopwords
    from string import punctuation

    stop_nltk = stopwords.words('english')
    stop_punctual = list(punctuation)
    stop_filter = ['no', 'not', 'won', 'don']
    for f in range(len(stop_filter)):
        stop_nltk.remove(stop_filter[f])

    stop_final = stop_nltk + stop_punctual + ["...", "``", "''", "====", "must"]
    stop_final = [t for t in text if t not in stop_final]

    return stop_final


def clean_text(text):
    text = [del_stop(t) for t in text]
    text = [' '.join(t) for t in text]
    return text


if __name__ == '__main__':
    from data import load_data

    reviews = load_data(config.TRAIN_FILE)
    reviews = remove_null_entries(reviews)
    reviews = text_to_list(reviews)
    reviews = normalise_text(reviews)
    reviews = tokenize_text(reviews)
    reviews = del_stop(reviews)
    reviews = clean_text(reviews)
    print(reviews)
