from collections import Counter
import pandas as pd
import numpy as np
import torch
import re
import unicodedata

MAPPING = {"positive": 1, "negative": 0}
MAX_LEN = 320
MIN_FREQ = 2
SEED = 42

def tokenize(text, max_len=MAX_LEN):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    tokens = [t.lower() for t in tokens]
    return tokens[:max_len]

def tokenize_full(text):
    return [t.lower() for t in re.findall(r"\w+|[^\w\s]", text, re.UNICODE)]

def tokenize_full(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return [t.lower() for t in tokens]

def build_train_word_counts(csv_path, seed=SEED, min_freq=MIN_FREQ):
    df0 = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    len80 = round(len(df0) * 0.8); mainTrain_df0 = df0.iloc[:len80]; len90 = round(len(mainTrain_df0) * 0.9)
    train_texts = mainTrain_df0.iloc[:len90]['review']
    counts = Counter(tok for text in train_texts for tok in tokenize_full(text))
    return Counter({w: c for w, c in counts.items() if c >= min_freq})

def build_vocab_from_counts(word_counts, reserve_n=2):
    reserved = {"<pad>", "<unk>"}
    ordered = sorted(((tok, c) for tok, c in word_counts.items() if tok not in reserved), key=lambda kv: (-kv[1], kv[0]))
    token2id = {"<pad>": 0, "<unk>": 1}
    token2id.update({tok: i for i, (tok, _) in enumerate(ordered, start=reserve_n)})
    id2token = {i: tok for tok, i in token2id.items()}
    return token2id, id2token

# -------

csv_path = "src/sentiment_ia_exo/data/IMDB_Dataset.csv"
df = pd.read_csv(csv_path)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df['review'] = df['review'].apply(tokenize)
df['sentiment'] = df['sentiment'].map(MAPPING)

len80 = round(len(df) * 0.8)
mainTrain_df = df.iloc[:len80]
len90 = round(len(mainTrain_df) * 0.9)

train_df = mainTrain_df.iloc[:len90]
val_df = mainTrain_df.iloc[len90:]

test_df = df.iloc[len80:]

x_train = train_df['review']
y_train = train_df['sentiment']

x_val = val_df['review']
y_val = val_df['sentiment']

x_test = test_df['review']
y_test = test_df['sentiment']

# -------

token2id, id2token = build_vocab_from_counts(build_train_word_counts(csv_path))
print(sorted(token2id.items()))