from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import math
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# ============================================
# Model Storage - Lazy Loading
# ============================================
models = {}
vectorizers = {}
tokenizers = {}

# Track which models are available
AVAILABLE_MODELS = []

# ============================================
# Load Logistic Regression (Always Available)
# ============================================
try:
    models["logreg"] = joblib.load("model.pkl")
    vectorizers["logreg"] = joblib.load("vectorizer.pkl")
    AVAILABLE_MODELS.append("logreg")
    print("Loaded: Logistic Regression")
except Exception as e:
    print(f"Failed to load Logistic Regression: {e}")

# ============================================
# Load Neural Network (TensorFlow/Keras)
# ============================================
try:
    import tensorflow as tf
    if os.path.exists("neural_network_model.keras"):
        models["neural"] = tf.keras.models.load_model("neural_network_model.keras")
        vectorizers["neural"] = joblib.load("nn_vectorizer.pkl")
        AVAILABLE_MODELS.append("neural")
        print("Loaded: Neural Network")
    else:
        print("Neural Network model file not found")
except Exception as e:
    print(f"Failed to load Neural Network: {e}")

# ============================================
# Load Naive Bayes (NLTK)
# ============================================
try:
    import pickle
    import nltk
    from nltk.stem.snowball import SnowballStemmer
    
    if os.path.exists("naive_bayes_model.pkl"):
        with open("naive_bayes_model.pkl", "rb") as f:
            models["naive_bayes"] = pickle.load(f)
        AVAILABLE_MODELS.append("naive_bayes")
        print("Loaded: Naive Bayes")
    else:
        print("Naive Bayes model file not found")
except Exception as e:
    print(f"Failed to load Naive Bayes: {e}")

# ============================================
# Load Transformer (HuggingFace PyTorch)
# ============================================
try:
    if os.path.exists("transformer_model"):
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        
        models["transformer"] = DistilBertForSequenceClassification.from_pretrained("transformer_model")
        models["transformer"].eval()  # Set to evaluation mode
        tokenizers["transformer"] = DistilBertTokenizerFast.from_pretrained("transformer_tokenizer")
        AVAILABLE_MODELS.append("transformer")
        print("Loaded: Transformer")
    else:
        print("Transformer model directory not found")
except Exception as e:
    print(f"Failed to load Transformer: {e}")

print(f"\nAvailable models: {AVAILABLE_MODELS}")

# ============================================
# Naive Bayes Featurization (matches notebook)
# ============================================
def featurize_for_naive_bayes(text):
    """Featurize text for Naive Bayes model (same as notebook)"""
    import nltk
    from nltk.stem.snowball import SnowballStemmer
    
    stemmer = SnowballStemmer("english")
    words = nltk.word_tokenize(text)
    features = defaultdict(int)
    all_caps = 0
    
    for word in words:
        if word.isupper():
            all_caps += 1
        stem = stemmer.stem(word.lower())
        features[stem] += 1
    
    if len(words) > 0:
        percent_caps = all_caps / len(words)
        features["CAPS AMOUNT"] = math.floor(percent_caps * 20)
    
    return dict(features)

# ============================================
# Prediction Functions
# ============================================
def predict_logreg(text):
    vector = vectorizers["logreg"].transform([text])
    pred = models["logreg"].predict(vector)[0]
    return str(pred)

def predict_neural(text):
    vector = vectorizers["neural"].transform([text]).toarray()
    probs = models["neural"].predict(vector, verbose=0).flatten()
    pred = 1 if probs[0] >= 0.5 else 0
    return str(pred)

def predict_naive_bayes(text):
    features = featurize_for_naive_bayes(text)
    pred = models["naive_bayes"].classify(features)
    # Convert "spam"/"ham" to 1/0
    return "1" if pred == "spam" else "0"

def predict_transformer(text):
    import torch
    
    encoding = tokenizers["transformer"](
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = models["transformer"](**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    
    return str(pred)

# ============================================
# API Routes
# ============================================
@app.route("/models", methods=["GET"])
def get_models():
    """Return list of available models"""
    model_info = {
        "logreg": {"name": "Logistic Regression", "description": "Fast, reliable linear classifier"},
        "neural": {"name": "Neural Network", "description": "Dense feed-forward classifier"},
        "naive_bayes": {"name": "Naive Bayes", "description": "Probabilistic classifier with stemming"},
        "transformer": {"name": "Transformer", "description": "DistilBERT deep learning model"}
    }
    
    available = []
    for model_id in AVAILABLE_MODELS:
        info = model_info.get(model_id, {"name": model_id, "description": ""})
        available.append({
            "id": model_id,
            "name": info["name"],
            "description": info["description"]
        })
    
    return jsonify({"models": available})

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "")
    model_type = request.json.get("model", "logreg")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if model_type not in AVAILABLE_MODELS:
        # Fallback to logreg if requested model not available
        if "logreg" in AVAILABLE_MODELS:
            model_type = "logreg"
        else:
            return jsonify({"error": f"Model '{model_type}' not available"}), 400
    
    try:
        if model_type == "logreg":
            pred = predict_logreg(text)
        elif model_type == "neural":
            pred = predict_neural(text)
        elif model_type == "naive_bayes":
            pred = predict_naive_bayes(text)
        elif model_type == "transformer":
            pred = predict_transformer(text)
        else:
            pred = predict_logreg(text)
        
        return jsonify({
            "prediction": pred,
            "model_used": model_type
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8000, debug=True)
