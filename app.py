from flask import Flask, render_template, request
from predict import predict_news

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news"]

    result, confidence = predict_news(news_text)

    return render_template(
        "result.html",
        news=news_text,
        result=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
