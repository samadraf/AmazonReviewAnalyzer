import pandas as pd


def extract_features(df, include_pos,default = False):
    df["char_length"] = df["text_"].apply(len)
    df["word_count"] = df["text_"].apply(len)
    df["punctuation_ct"] =