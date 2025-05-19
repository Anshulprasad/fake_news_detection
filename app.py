from config import MODEL_BERT_PATH, TOKENIZER_BERT_PATH, MODEL_NN_PATH, VECTORIZER_PATH, MODEL_MNB_PATH
import joblib
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_BERT_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_BERT_PATH)

# temperory
model = joblib.load(MODEL_MNB_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict whether a news article is fake or real.
    """
    data = request.get_json()  # Get JSON data from the request
    news_text = data.get("text", "")  # Extract the 'text' field

    if not news_text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize the input text
    inputs = tokenizer(
        news_text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # temperory
    news_vector = vectorizer.transform([news_text])
    prediction = model.predict(news_vector)[0]
    
    # Map the prediction to a label
    result = "Real" if prediction == 1 else "Fake"

    # Return the prediction as JSON
    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
