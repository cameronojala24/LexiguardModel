# Lexiguard

A machine learning–powered text spam detection system built for the final NLP project.
Lexiguard includes multiple classifiers, a Flask API server, and a React-based web client for real-time predictions.

## Deployment

The application is currently deployed on Render: https://lexiguardnlp.onrender.com/

**Note:** Due to free tier memory constraints (512MB), the deployed version only runs Logistic Regression. For full 4-model functionality, run locally.

## Models Included

| Model | Description |
|---|---|
| Logistic Regression | Reliable linear classifier |
| Naive Bayes | Probabilistic classifier with stemming |
| Neural Network | Dense feed-forward classifier |
| Transformer | DistilBERT fine-tuned model (99.28% accuracy) |

The Flask API supports all 4 models with dynamic switching via dropdown selector.

## Steps to run:

### Prerequisites

If pulling for the first time, install Git LFS to download the Transformer model:

```bash
brew install git-lfs
git lfs install
git lfs pull
```

### Env

Create a `.env` file in `client/` with:

```
REACT_APP_API_URL=http://localhost:8000
```

### Server

```bash
cd server
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
python app.py
```

The backend will start on: http://localhost:8000

### Client

```bash
cd client
npm install
npm start
```

The frontend will start on: http://localhost:3000
