# Lexiguard

A machine learning–powered text spam detection system built for the final NLP project.
Lexiguard includes multiple classifiers, a Flask API server, and a React-based web client for real-time predictions.

## Deployment
The application is currently deployed on Render: https://lexiguardnlp.onrender.com/.

## Models Included

The `models/` directory contains three trained classifiers:

| Model | Description |
|---|---|
| Naive Bayes | Fast, interpretable baseline classifier |
| Logistic Regression | Reliable linear classifier |
| Neural Network | Basic dense feed-forward classifier |
| Transformer | More complex NN with transformer architecture |


The Flask API currently deploys the **logistic regression model**, with the vectorizer and model saved in the `server/` folder.

## Steps to run:
### Env
To run locally, create a .env file in client with:

REACT_APP_API_URL=http://localhost:8000

### Server

```bash
cd server
python app.py
```

The backend will start on:
http://localhost:8000

### Client

```bash
cd client
npm start
```

The frontend will start on:
http://localhost:3000
