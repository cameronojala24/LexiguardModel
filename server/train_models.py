#!/usr/bin/env python3
"""
Train and export all models for Lexiguard
This script will download the dataset and train all 4 models
"""

import os
import csv
import math
import pickle
import joblib
import numpy as np
from collections import defaultdict

print("=" * 50)
print("LEXIGUARD MODEL TRAINER")
print("=" * 50)

# ============================================
# 1. Download Dataset
# ============================================
print("\n[1/5] Downloading dataset...")

try:
    import kagglehub
    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    dataset_path = os.path.join(path, "spam.csv")
    print(f"Dataset path: {dataset_path}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Trying alternative method...")
    # Try to find existing dataset
    possible_paths = [
        os.path.expanduser("~/.cache/kagglehub/datasets/uciml/sms-spam-collection-dataset/versions/1/spam.csv"),
        "spam.csv",
        "../models/spam.csv"
    ]
    dataset_path = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = p
            break
    if not dataset_path:
        print("ERROR: Could not find dataset. Please download spam.csv manually.")
        exit(1)

# Load dataset
print("\n[2/5] Loading dataset...")
with open(dataset_path, 'r', encoding='latin-1') as f:
    reader = csv.reader(f)
    rows = list(reader)

if rows[0][0].lower() == "v1":
    rows = rows[1:]

data = np.array([[r[0], r[1]] for r in rows], dtype=object)
labels_raw = data[:, 0]
messages = data[:, 1]

# Convert labels
labels = np.where(labels_raw == "spam", 1, 0)
print(f"Loaded {len(messages)} messages")
print(f"Spam: {sum(labels)}, Ham: {len(labels) - sum(labels)}")

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================
# 3. Train Logistic Regression (already done, but let's verify)
# ============================================
print("\n[3/5] Training Logistic Regression...")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

vectorizer_logreg = CountVectorizer()
X_train_vec = vectorizer_logreg.fit_transform(X_train)
X_test_vec = vectorizer_logreg.transform(X_test)

model_logreg = LogisticRegression(max_iter=200, solver="liblinear")
model_logreg.fit(X_train_vec, y_train)

preds = model_logreg.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
print(f"Logistic Regression Accuracy: {acc:.4f}")

# Save (overwrite existing to ensure consistency)
joblib.dump(model_logreg, "model.pkl")
joblib.dump(vectorizer_logreg, "vectorizer.pkl")
print("Saved: model.pkl, vectorizer.pkl")

# ============================================
# 4. Train Neural Network
# ============================================
print("\n[4/5] Training Neural Network...")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    # Use same vectorizer format
    vectorizer_nn = CountVectorizer()
    X_train_nn = vectorizer_nn.fit_transform(X_train).toarray()
    X_test_nn = vectorizer_nn.transform(X_test).toarray()
    
    input_dim = X_train_nn.shape[1]
    
    model_nn = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model_nn.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train
    model_nn.fit(
        X_train_nn, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    probs = model_nn.predict(X_test_nn, verbose=0).flatten()
    preds_nn = (probs >= 0.5).astype(int)
    acc_nn = accuracy_score(y_test, preds_nn)
    print(f"Neural Network Accuracy: {acc_nn:.4f}")
    
    # Save
    model_nn.save("neural_network_model.keras")
    joblib.dump(vectorizer_nn, "nn_vectorizer.pkl")
    print("Saved: neural_network_model.keras, nn_vectorizer.pkl")
    
except ImportError:
    print("TensorFlow not installed, skipping Neural Network")
except Exception as e:
    print(f"Error training Neural Network: {e}")

# ============================================
# 5. Train Naive Bayes
# ============================================
print("\n[5/5] Training Naive Bayes...")

try:
    import nltk
    from nltk.classify import NaiveBayesClassifier
    from nltk.stem.snowball import SnowballStemmer
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    def featurize(words):
        """Featurize text for Naive Bayes"""
        stemmer = SnowballStemmer("english")
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
    
    # Prepare data for NLTK
    train_data = []
    for i, msg in enumerate(X_train):
        words = nltk.word_tokenize(msg)
        label = "spam" if y_train[i] == 1 else "ham"
        train_data.append((featurize(words), label))
    
    # Train
    model_nb = NaiveBayesClassifier.train(train_data)
    
    # Evaluate
    correct = 0
    for i, msg in enumerate(X_test):
        words = nltk.word_tokenize(msg)
        pred = model_nb.classify(featurize(words))
        actual = "spam" if y_test[i] == 1 else "ham"
        if pred == actual:
            correct += 1
    
    acc_nb = correct / len(X_test)
    print(f"Naive Bayes Accuracy: {acc_nb:.4f}")
    
    # Save
    with open("naive_bayes_model.pkl", "wb") as f:
        pickle.dump(model_nb, f)
    print("Saved: naive_bayes_model.pkl")
    
except ImportError:
    print("NLTK not installed, skipping Naive Bayes")
except Exception as e:
    print(f"Error training Naive Bayes: {e}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)

# Check what was saved
saved_files = [
    ("model.pkl", "Logistic Regression"),
    ("vectorizer.pkl", "LogReg Vectorizer"),
    ("neural_network_model.keras", "Neural Network"),
    ("nn_vectorizer.pkl", "NN Vectorizer"),
    ("naive_bayes_model.pkl", "Naive Bayes"),
]

print("\nSaved models:")
for filename, desc in saved_files:
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"  ✓ {filename} ({size:,} bytes) - {desc}")
    else:
        print(f"  ✗ {filename} - NOT CREATED")

print("\nNote: Transformer model requires HuggingFace and takes ~30 min to train.")
print("Skipping Transformer for now.")
print("\nRestart the server to load new models!")

