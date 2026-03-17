from flask import Flask, render_template, request
from predict import predict_news

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")



@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']

    result = predict_news(news_text)

    return f"<h2>Prediction: {result}</h2><br><a href='/'>Go Back</a>"


if __name__ == "__main__":
    app.run(debug=True)
