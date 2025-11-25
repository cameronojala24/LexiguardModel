from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    
    vector = vectorizer.transform([text])
    pred = model.predict(vector)[0]

    return jsonify({"prediction": str(pred)})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
