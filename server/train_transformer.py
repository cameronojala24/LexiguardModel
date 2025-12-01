#!/usr/bin/env python3
"""
Train and export Transformer (DistilBERT) model for Lexiguard
This takes ~20-30 minutes on CPU
Uses PyTorch backend (more stable than TensorFlow)
"""

import os
import csv
import numpy as np

print("=" * 50)
print("LEXIGUARD TRANSFORMER TRAINER")
print("=" * 50)

# ============================================
# 1. Load dependencies
# ============================================
print("\n[1/6] Loading dependencies...")

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("PyTorch version:", torch.__version__)
print("Device:", "MPS (Apple Silicon)" if torch.backends.mps.is_available() else "CPU")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================
# 2. Load Dataset
# ============================================
print("\n[2/6] Loading dataset...")

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
    print("Downloading dataset...")
    import kagglehub
    path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
    dataset_path = os.path.join(path, "spam.csv")

print(f"Using dataset: {dataset_path}")

with open(dataset_path, 'r', encoding='latin-1') as f:
    reader = csv.reader(f)
    rows = list(reader)

if rows[0][0].lower() == "v1":
    rows = rows[1:]

data = np.array([[r[0], r[1]] for r in rows], dtype=object)
labels_raw = data[:, 0]
messages = data[:, 1].tolist()

labels = [1 if l == "spam" else 0 for l in labels_raw]
print(f"Loaded {len(messages)} messages")

X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================
# 3. Load Tokenizer and Model
# ============================================
print("\n[3/6] Loading DistilBERT tokenizer and model...")
print("(This downloads ~250MB on first run)")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)
model.to(device)

print("Model loaded!")

# ============================================
# 4. Tokenize Data
# ============================================
print("\n[4/6] Tokenizing data...")

train_encodings = tokenizer(
    X_train,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

test_encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SpamDataset(train_encodings, y_train)
test_dataset = SpamDataset(test_encodings, y_test)

print(f"Created datasets")

# ============================================
# 5. Train Model
# ============================================
print("\n[5/6] Training model...")
print("This will take 15-30 minutes depending on hardware...")
print("-" * 50)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./transformer_training",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    use_mps_device=torch.backends.mps.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ============================================
# 6. Evaluate and Save
# ============================================
print("\n[6/6] Evaluating and saving model...")

results = trainer.evaluate()
print(f"Final Accuracy: {results['eval_accuracy']:.4f}")

print("\nSaving model to transformer_model/...")
model.save_pretrained("transformer_model")
tokenizer.save_pretrained("transformer_tokenizer")

print("\n" + "=" * 50)
print("TRANSFORMER TRAINING COMPLETE!")
print("=" * 50)

if os.path.exists("transformer_model") and os.path.exists("transformer_tokenizer"):
    print("\n Saved: transformer_model/")
    print(" Saved: transformer_tokenizer/")
    print(f"\nAccuracy: {results['eval_accuracy']:.2%}")
    print("\nRestart the server to load the Transformer model!")
else:
    print("\nError: Model not saved correctly")
