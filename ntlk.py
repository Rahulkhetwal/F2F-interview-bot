## Classifying user intents using NLTK for preprocessing and a deep learning model (specifically a simple feedforward neural network) for classification. 
## used a simpler count vectorizer instead of the TF-IDF vectorizer, and a simpler feedforward neural network model with only one hidden layer.
## No tokenization and lowercasing


import nltk
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Sample dataset of user queries and intents
data = [
    ("How can I make a reservation?", "reservation"),
    ("What time does the restaurant close?", "opening_hours"),
    # Add more examples...
]

# Preprocess the data
preprocessed_data = [(nltk.word_tokenize(text.lower()), intent) for text, intent in data]

# Split data into features and labels
X = [" ".join(tokens) for tokens, _ in preprocessed_data]
y = [intent for _, intent in preprocessed_data]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define a count vectorizer
count_vectorizer = CountVectorizer()

# Fit and transform count vectorizer on training data
X_train_count = count_vectorizer.fit_transform(X_train)

# Transform testing data
X_test_count = count_vectorizer.transform(X_test)

# Define a simple feedforward neural network model
model = Sequential()
model.add(Dense(64, input_shape=(X_train_count.shape[1],), activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_count, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_count, y_test)
print("Test Accuracy:", accuracy)

# Predict on new data
new_queries = ["Can I book a table for tonight?", "What are the restaurant's opening hours?"]
preprocessed_new_queries = [" ".join(nltk.word_tokenize(text.lower())) for text in new_queries]
X_new_count = count_vectorizer.transform(preprocessed_new_queries)
predictions = model.predict(X_new_count)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
print("Predicted Intents:", predicted_labels)