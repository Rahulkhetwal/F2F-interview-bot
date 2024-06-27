import joblib

# Load the trained components from disk
model = joblib.load("trained_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Preprocess the new data (tokenization, lemmatization, etc.)
# Split the new data into features (X_new)
X_new = ["new user query"]

# Transform the new data using the trained TF-IDF vectorizer
X_new_tfidf = tfidf_vectorizer.transform(X_new)

# Make predictions on the new data using the trained model
y_pred = model.predict(X_new_tfidf)

# Decode the predicted label using the trained label encoder
y_pred_label = label_encoder.inverse_transform(y_pred)

# Print the predicted label
print("Predicted label:", y_pred_label[0])