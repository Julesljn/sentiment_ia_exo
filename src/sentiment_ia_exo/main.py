import pandas as pd
import numpy as np
import torch

mapping = {"positive": 1, "negative": 0}

csv_path = "src/sentiment_ia_exo/data/IMDB_Dataset.csv"
df = pd.read_csv(csv_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['sentiment'] = df['sentiment'].map(mapping)

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