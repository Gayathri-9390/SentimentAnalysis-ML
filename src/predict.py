from src.preprocess import clean_text
import pickle

# Load the saved model
with open("models/sentiment_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

def predict_sentiment(review):
    review_clean = clean_text(review)
    
    # Handle empty input
    if not review_clean:
        return "empty"
    
    # Simple keyword check for very short reviews
    if review_clean in ["good", "excellent", "amazing", "perfect", "love"]:
        return "positive"
    if review_clean in ["bad", "terrible", "poor", "horrible", "disappointing"]:
        return "negative"
    
    # ML prediction
    review_vec = vectorizer.transform([review_clean])
    return model.predict(review_vec)[0]

# Interactive loop
print("=== Sentiment Analysis ===")
print("Type multiple reviews. Press Enter on an empty line to predict all. Type 'exit' to quit.\n")

while True:
    print("Enter your reviews (empty line to process, 'exit' to quit):")
    reviews = []
    while True:
        line = input()
        line = line.strip()  # remove extra spaces
        if line.lower() == "exit":
            exit()          # exit program
        if not line:        # empty line means stop collecting reviews
            break
        reviews.append(line)

    if not reviews:
        continue  # no reviews entered, ask again

    for review in reviews:
        sentiment = predict_sentiment(review)
        print(f"Review: {review}\nPredicted sentiment: {sentiment}\n")

