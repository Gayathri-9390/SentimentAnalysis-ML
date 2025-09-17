import tkinter as tk
from tkinter import scrolledtext, messagebox
import pickle
import os
import sys
import csv

# Add current folder to path
sys.path.append(os.path.dirname(__file__))

# Import preprocess.py
from preprocess import clean_text

# Load saved model
model_path = os.path.join(os.path.dirname(__file__), "../models/sentiment_model.pkl")
with open(model_path, "rb") as f:
    vectorizer, model = pickle.load(f)

# Keywords for short reviews
POSITIVE_WORDS = ["good", "excellent", "amazing", "perfect", "love", "great"]
NEGATIVE_WORDS = ["bad", "terrible", "poor", "horrible", "disappointing", "awful"]

def predict_sentiment(review):
    review_clean = clean_text(review)
    
    if not review_clean:
        return "empty"
    
    words = review_clean.split()
    if any(word in POSITIVE_WORDS for word in words):
        return "positive"
    if any(word in NEGATIVE_WORDS for word in words):
        return "negative"
    
    review_vec = vectorizer.transform([review_clean])
    return model.predict(review_vec)[0]

def predict_reviews():
    input_text = input_area.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Input Required", "Please enter at least one review.")
        return
    
    reviews = input_text.split("\n")
    output_area.config(state='normal')
    output_area.delete("1.0", tk.END)
    
    global last_predictions
    last_predictions = []
    
    for review in reviews:
        review = review.strip()
        if review:
            sentiment = predict_sentiment(review)
            output_area.insert(tk.END, f"Review: {review}\nPredicted sentiment: {sentiment}\n\n")
            last_predictions.append((review, sentiment))
    
    output_area.config(state='disabled')

def save_predictions():
    if not last_predictions:
        messagebox.showwarning("No Predictions", "No predictions to save.")
        return
    
    save_path = os.path.join(os.path.dirname(__file__), "predictions.csv")
    with open(save_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Review", "Predicted Sentiment"])
        for review, sentiment in last_predictions:
            writer.writerow([review, sentiment])
    
    messagebox.showinfo("Saved", f"Predictions saved to {save_path}")

# Tkinter window
window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry("650x550")
window.configure(bg="#f0f0f0")

# Labels
tk.Label(window, text="Enter reviews (one per line):", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=5)

# Input box
input_area = scrolledtext.ScrolledText(window, height=10, bg="#ffffff", fg="#000000", font=("Arial", 11))
input_area.pack(fill=tk.BOTH, padx=10, pady=5)

# Buttons
predict_button = tk.Button(window, text="Predict Sentiment", command=predict_reviews, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
predict_button.pack(pady=5)

save_button = tk.Button(window, text="Save Predictions", command=save_predictions, bg="#2196F3", fg="white", font=("Arial", 11, "bold"))
save_button.pack(pady=5)

# Output box
tk.Label(window, text="Predictions:", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=5)

output_area = scrolledtext.ScrolledText(window, height=15, bg="#f9f9f9", fg="#333333", font=("Arial", 11), state='disabled')
output_area.pack(fill=tk.BOTH, padx=10, pady=5)

# Store last predictions
last_predictions = []

window.mainloop()
