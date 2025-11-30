import { useState } from "react";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL;

function App() {
  const [text, setText] = useState("");
  const [prediction, setPrediction] = useState("");

  const handleSubmit = async () => {
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      console.error("Error calling prediction API:", err);
      setPrediction("Error contacting server");
    }
  };

  return (
    <div className="App">
      <h2>Spam Text Classifier</h2>

      <textarea
        rows="5"
        cols="50"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text here..."
      />

      <br />

      <button onClick={handleSubmit}>Predict</button>

      {prediction !== "" && (
        <div
          style={{
            marginTop: "24px",
            padding: "16px",
            borderRadius: "8px",
            borderLeft: prediction === "1" ? "4px solid #d72638" : "4px solid #2a9d8f",
            background: prediction === "1" ? "#ffe6e9" : "#e8fff7",
            color: prediction === "1" ? "#8b0f1c" : "#145c3a"
          }}
        >
          <strong>Prediction:</strong>{" "}
          {prediction === "1" ? "Spam ❌" : "Not Spam ✔️"}
        </div>
      )}
    </div>
  );
}

export default App;
