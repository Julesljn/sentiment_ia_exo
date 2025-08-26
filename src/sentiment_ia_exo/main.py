import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

csv_path = "src/sentiment_ia_exo/data/IMDB_Dataset.csv"
df = pd.read_csv(csv_path)

len80 = round(len(df) * 80 / 100)
len20 = round(len(df) * 20 / 100)


train_text = df['review']
print(len80, len20)