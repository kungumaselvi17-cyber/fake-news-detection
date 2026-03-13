import pickle
from utils import clean_text

model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def predict_news(news_text):
    news_text = clean_text(news_text)
    vector = vectorizer.transform([news_text])
    result = model.predict(vector)[0]

    if result == 1:
        return "REAL NEWS ✅"
    else:
        return "FAKE NEWS ❌"
