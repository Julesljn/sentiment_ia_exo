import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

csv_path = "src/sentiment_ia_exo/data/IMDB_Dataset.csv"
df = pd.read_csv(csv_path)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

len80 = round(len(df) * 0.8)
train_df = df.iloc[:len80]
test_df  = df.iloc[len80:]


x_train = train_df['review']
y_train = train_df['sentiment']

x_test = test_df['review']
y_test = test_df['sentiment']

