from predict import predict_news

print("📰 Fake News Detection System")
print("--------------------------------")

while True:
    news = input("Enter news text: ")

    result, confidence = predict_news(news)

    print("Prediction:", result)
    print("Confidence:", confidence, "%")

    choice = input("Check another? (y/n): ")
    if choice.lower() != "y":
        break
