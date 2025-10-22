import pandas as pd
import string
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
POS_WHITELIST = {"VERB", "NOUN", "ADV"}

def extract_features(df, include_pos,default = False):
    df['char_length'] = df['cleaned_text'].apply(len)
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df['punctuation_ct'] = df['cleaned_text'].apply(
        lambda x: sum(1 for char in x if char in string.punctuation)
    )
    df['is_extreme_star'] = df['rating'].isin([1.0, 5.0]).astype(int)
    if include_pos:
        df = add_pos_features(df)

    return df

def pos_counts(text):
    doc = nlp(text)
    return Counter(token.pos_ for token in doc if token.pos_ in POS_WHITELIST)

def add_pos_features(df):
    pos_data = df["cleaned_text"].apply(pos_counts)
    pos_df = pd.DataFrame(list(pos_data)).fillna(0)
    pos_df.index = df.index  # Align new POS DataFrame with original
    return pd.concat([df, pos_df], axis=1)