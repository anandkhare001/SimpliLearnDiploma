import pandas as pd
import numpy as np
import os
import re


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

