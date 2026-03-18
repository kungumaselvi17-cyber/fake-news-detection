import pickle
from utils import clean_text


# -------------------------------
# Load Model & Vectorizer
# -------------------------------
try:
    with open("model/fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    print("Error loading model files:", e)


# -------------------------------
# Prediction Function
# -------------------------------
def predict_news(news_text):
    """
    Predict whether news is REAL or FAKE
    Returns:
        label (str)
        confidence (float)
    """

    if not news_text.strip():
        return "No text provided ❗", 0

    # Step 1: Clean text
    cleaned_text = clean_text(news_text)

    # Step 2: Text → Vector
    vector = vectorizer.transform([cleaned_text])

    # Step 3: Prediction
    prediction = model.predict(vector)[0]

    # Step 4: Confidence Score
    try:
        confidence = model.predict_proba(vector).max() * 100
    except:
        confidence = 0

    # Step 5: Result Label
    if prediction == 1:
        label = "REAL NEWS ✅"
    else:
        label = "FAKE NEWS ❌"

    return label, round(confidence, 2)
