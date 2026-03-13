import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import clean_text

# Load dataset
data = pd.read_csv("dataset/news.csv")

# Clean text
data["text"] = data["text"].apply(clean_text)

# Convert labels
data["label"] = data["label"].map({"REAL":1, "FAKE":0})

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")
