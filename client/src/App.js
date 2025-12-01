import { useState, useCallback, useEffect } from "react";
import "./App.css";
import PageLoader from "./PageLoader";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// Translations for English and Spanish
const translations = {
  en: {
    title: "Lexiguard",
    subtitle: "Spam Detection for La Alianza",
    placeholder: "Enter a message to analyze...",
    button: "Analyze Message",
    loading: "Analyzing...",
    spamResult: "This looks like spam",
    hamResult: "This message looks safe",
    examplesLabel: "Try an example:",
    modelLabel: "Model:",
    langToggle: "ES",
    errorMessage: "Error contacting server. Please try again.",
  },
  es: {
    title: "Lexiguard",
    subtitle: "Detección de Spam para La Alianza",
    placeholder: "Ingrese un mensaje para analizar...",
    button: "Analizar Mensaje",
    loading: "Analizando...",
    spamResult: "Esto parece ser spam",
    hamResult: "Este mensaje parece seguro",
    examplesLabel: "Prueba un ejemplo:",
    modelLabel: "Modelo:",
    langToggle: "EN",
    errorMessage: "Error al contactar el servidor. Inténtelo de nuevo.",
  },
};

// Fallback model options if API doesn't respond
const fallbackModels = [
  { id: "logreg", name: "Logistic Regression", description: "Fast, reliable linear classifier" }
];

// Example messages for demo
const exampleMessages = [
  {
    text: "CONGRATULATIONS! You've won a FREE iPhone! Click here to claim your prize NOW!!!",
    type: "spam",
  },
  {
    text: "Hey, are we still meeting for coffee tomorrow at 3pm?",
    type: "ham",
  },
  {
    text: "ALERT! You've been selected for a FREE GIFT PACK! Tap the link to claim it RIGHT NOW!!!",
    type: "spam",
  },
  {
    text: "Mom said dinner is at 6. Don't forget to bring the salad!",
    type: "ham",
  },
];

function App() {
  const [text, setText] = useState("");
  const [prediction, setPrediction] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [language, setLanguage] = useState("en");
  const [showUI, setShowUI] = useState(false);
  const [loaderDone, setLoaderDone] = useState(false);
  
  // Model selection state
  const [selectedModel, setSelectedModel] = useState("logreg");
  const [availableModels, setAvailableModels] = useState(fallbackModels);
  const [modelUsed, setModelUsed] = useState("");

  // Fetch available models from API
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${API_URL}/models`);
        const data = await response.json();
        if (data.models && data.models.length > 0) {
          setAvailableModels(data.models);
          // Select first available model
          setSelectedModel(data.models[0].id);
        }
      } catch (err) {
        console.log("Could not fetch models, using fallback");
      }
    };
    fetchModels();
  }, []);

  // Called when text has split enough to reveal UI
  const handleReveal = useCallback(() => {
    setShowUI(true);
  }, []);

  // Called when loader is completely done
  const handleLoaderComplete = useCallback(() => {
    setLoaderDone(true);
  }, []);

  const t = translations[language];

  const handleSubmit = async () => {
    if (!text.trim()) return;
    
    setIsLoading(true);
    setPrediction("");
    setModelUsed("");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: selectedModel }),
      });

      const data = await response.json();
      setPrediction(data.prediction);
      setModelUsed(data.model_used || selectedModel);
    } catch (err) {
      console.error("Error calling prediction API:", err);
      setPrediction("error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (exampleText) => {
    setText(exampleText);
    setPrediction("");
    setModelUsed("");
  };

  const toggleLanguage = () => {
    setLanguage((prev) => (prev === "en" ? "es" : "en"));
  };

  const getPredictionDisplay = () => {
    if (prediction === "error") {
      return {
        text: t.errorMessage,
        className: "prediction-result error",
        icon: (
          <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
        ),
      };
    }
    if (prediction === "1") {
      return {
        text: t.spamResult,
        className: "prediction-result spam",
        icon: (
          <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        ),
      };
    }
    return {
      text: t.hamResult,
      className: "prediction-result safe",
      icon: (
        <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
      ),
    };
  };

  return (
    <>
      {!loaderDone && <PageLoader onReveal={handleReveal} onComplete={handleLoaderComplete} />}
      <div className={`app-container ${showUI ? "visible" : ""} ${loaderDone ? "loaded" : ""}`}>
        <div className={`app-card ${showUI ? "visible" : ""} ${loaderDone ? "loaded" : ""}`}>
          {/* GitHub Link */}
          <a
            href="https://github.com/cameronojala24/LexiguardNLP"
            target="_blank"
            rel="noopener noreferrer"
            className="github-link"
            aria-label="View source on GitHub"
          >
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
            </svg>
          </a>

          {/* Language Toggle */}
          <button className="lang-toggle" onClick={toggleLanguage}>
            {t.langToggle}
          </button>

          {/* Header */}
          <header className="app-header">
            <h1 className="app-title">{t.title}</h1>
            <p className="app-subtitle">{t.subtitle}</p>
          </header>

          {/* Model Selector */}
          <div className="model-selector">
            <label className="model-label">{t.modelLabel}</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="model-dropdown"
            >
              {availableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          {/* Example Messages */}
          <div className="examples-section">
            <span className="examples-label">{t.examplesLabel}</span>
            <div className="examples-grid">
              {exampleMessages.map((example, index) => (
                <button
                  key={index}
                  className={`example-chip ${example.type}`}
                  onClick={() => handleExampleClick(example.text)}
                >
                  {example.text.length > 40
                    ? example.text.substring(0, 40) + "..."
                    : example.text}
                </button>
              ))}
            </div>
          </div>

          {/* Input Area */}
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={t.placeholder}
            className="message-input"
          />

          {/* Submit Button */}
          <button
            className={`submit-button ${isLoading ? "loading" : ""}`}
            onClick={handleSubmit}
            disabled={isLoading || !text.trim()}
          >
            {isLoading ? t.loading : t.button}
          </button>

          {/* Prediction Result */}
          {prediction !== "" && (() => {
            const display = getPredictionDisplay();
            return (
              <div className={display.className}>
                {display.icon && (
                  <span className="result-icon">{display.icon}</span>
                )}
                <span className="result-text">{display.text}</span>
                {modelUsed && prediction !== "error" && (
                  <span className="model-badge">
                    {availableModels.find(m => m.id === modelUsed)?.name || modelUsed}
                  </span>
                )}
              </div>
            );
          })()}
        </div>
      </div>
    </>
  );
}

export default App;
