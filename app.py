from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and preprocessing components
model = joblib.load("model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/classify_intent", methods=["POST"])
def classify_intent():
    # Get input text from request
    data = request.get_json()
    text = data["text"]

    # Vectorize preprocessed text
    X = tfidf_vectorizer.transform([text])

    # Predict intent using the trained model
    predicted_intent_index = model.predict(X)[0]
    predicted_intent = label_encoder.inverse_transform([predicted_intent_index])[0]

    return jsonify({"intent": predicted_intent})

if __name__ == "__main__":
    app.run(debug=True)