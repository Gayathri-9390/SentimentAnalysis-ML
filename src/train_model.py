import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
from src.preprocess import clean_text
import os

data = pd.read_csv("data/reviews.csv")
data['cleaned_review'] = data['review'].apply(clean_text)

X = data['cleaned_review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Ensure models folder exists
os.makedirs('models', exist_ok=True)
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
