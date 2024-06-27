import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import os

# Sample dataset of user queries and intents
data = [
    ("How can I make a reservation?", "reservation"),
    ("What time does the restaurant close?", "opening_hours"),
    # Add more examples...
]

# Preprocess the data (tokenization, lemmatization, etc.)
# Split the data into features (X) and labels (y)
X = [text for text, _ in data]
y = [intent for _, intent in data]

# Train a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train a label encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train a machine learning model (e.g., SVM)
model = SVC(kernel='linear')
model.fit(X_tfidf, y_encoded)

# Save the trained components to disk
joblib.dump(model, "trained_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")


