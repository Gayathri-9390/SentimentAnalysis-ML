# 📝 Sentiment Analysis on Product Reviews

A Python-based project that predicts the sentiment of product reviews (positive, negative, or neutral) using Machine Learning. The project includes both a **GUI** and **CLI** interface for interactive sentiment prediction.

---

## ⚡ Features

- 🏋️ Train a machine learning model (Logistic Regression with TF-IDF) on product reviews.  
- 🖥️ Predict sentiment via **CLI** or **GUI**.  
- ✨ Handles **short reviews**, **multiple reviews**, and **empty lines**.  
- 💾 Save predictions to a **CSV file**.  
- 🎨 User-friendly GUI with enhanced **colors** and **fonts** for better experience.

---

## 📂 Project Structure

SentimentAnalysis-ML/
├── src/

│ ├── preprocess.py

│ ├── train_model.py

│ ├── predict.py

│ ├── predict_gui.py

│ └── init.py

├── data/ # Dataset of product reviews

├── models/ # Trained ML model (sentiment_model.pkl)

├── requirements.txt

└── README.md



---

## ⚡ Setup Instructions

### 📂 Clone the repository:

git clone https://github.com/Gayathri-9390/SentimentAnalysis-ML.git
cd SentimentAnalysis-ML


📦 Install dependencies:
 
pip install -r requirements.txt

🏋️ Train the model (optional if you already have sentiment_model.pkl):

python src/train_model.py


🖥️ Run the GUI:

python src/predict_gui.py
Enter one or more reviews in the top box (one per line).

Click Predict Sentiment.

View predictions in the bottom box.

Click Save Predictions to save results to predictions.csv.

---

📸 Screenshots:

(Optional: Add a screenshot of your GUI here to showcase your interface)

---

🖱️ Usage Example (CLI):

python src/predict.py
Type reviews interactively.

Get immediate sentiment prediction.

---

🛠️ Technologies Used:

🐍 Python 3.x

🤖 Scikit-learn (Machine Learning)

🖼️ Tkinter (GUI)

📊 Pandas / CSV for data handling

---

🚀 Future Work:

📈 Expand dataset – Include more product reviews to improve model accuracy.

⚡ Advanced models – Experiment with other ML algorithms (Naive Bayes, SVM, or deep learning models) for better predictions.

📊 Sentiment visualization – Add charts/graphs to visualize positive, negative, and neutral review distribution.

🎨 GUI enhancements – Add themes, icons, or interactive features for better user experience.

🌐 Multi-language support – Extend the system to handle reviews in multiple languages.

💻 Web version – Deploy the project as a web app for wider accessibility.

---

📝 Notes:

The model can be improved by adding more labeled reviews.

Short reviews like "good" or "bad" are handled via keyword matching.

The GUI is designed to be user-friendly with colors, fonts, and save functionality.

---

📄 License:

This project is open-source and free to use.