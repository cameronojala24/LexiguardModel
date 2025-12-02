import { useState, useCallback, useEffect } from "react";
import "./App.css";
import PageLoader from "./PageLoader";
import Select from "react-select";

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
    // Digital Literacy Section
    warningSignsTitle: "⚠️ Warning Signs Detected",
    learnMoreTitle: "📚 Learn to Spot Spam",
    whySpamTitle: "Why might this be spam?",
    tipsTitle: "Tips to Stay Safe",
    indicators: {
      urgency: { label: "Creates urgency", tip: "Scammers pressure you to act fast so you don't think clearly" },
      freePrize: { label: "Offers free prizes", tip: "If you didn't enter a contest, you can't win one" },
      allCaps: { label: "Uses ALL CAPS", tip: "Legitimate businesses don't shout at you" },
      clickLink: { label: "Asks to click links", tip: "Never click links from unknown senders" },
      personalInfo: { label: "Requests personal info", tip: "Banks and companies never ask for passwords via text" },
      moneyRequest: { label: "Mentions money/prizes", tip: "Free money offers are almost always scams" },
      suspiciousNumbers: { label: "Contains suspicious numbers", tip: "Random phone numbers or codes are often fake" },
      actionWords: { label: "Uses pressure words", tip: "Words like 'immediately', 'now', 'urgent' are red flags" },
    },
    safetyTips: [
      "Never share passwords, PINs, or Social Security numbers",
      "Call the company directly using a number you trust",
      "Don't click links in unexpected messages",
      "If it sounds too good to be true, it probably is",
      "Ask a family member or friend if you're unsure",
    ],
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
    // Digital Literacy Section
    warningSignsTitle: "⚠️ Señales de Advertencia Detectadas",
    learnMoreTitle: "📚 Aprende a Identificar Spam",
    whySpamTitle: "¿Por qué esto podría ser spam?",
    tipsTitle: "Consejos de Seguridad",
    indicators: {
      urgency: { label: "Crea urgencia", tip: "Los estafadores te presionan para que actúes rápido sin pensar" },
      freePrize: { label: "Ofrece premios gratis", tip: "Si no participaste en un concurso, no puedes ganar" },
      allCaps: { label: "Usa MAYÚSCULAS", tip: "Las empresas legítimas no te gritan" },
      clickLink: { label: "Pide hacer clic en enlaces", tip: "Nunca hagas clic en enlaces de remitentes desconocidos" },
      personalInfo: { label: "Pide información personal", tip: "Los bancos nunca piden contraseñas por mensaje" },
      moneyRequest: { label: "Menciona dinero/premios", tip: "Las ofertas de dinero gratis casi siempre son estafas" },
      suspiciousNumbers: { label: "Contiene números sospechosos", tip: "Números de teléfono o códigos aleatorios suelen ser falsos" },
      actionWords: { label: "Usa palabras de presión", tip: "Palabras como 'inmediatamente', 'ahora', 'urgente' son señales de alerta" },
    },
    safetyTips: [
      "Nunca comparta contraseñas, PINs o números de seguro social",
      "Llame a la empresa directamente usando un número de confianza",
      "No haga clic en enlaces de mensajes inesperados",
      "Si suena demasiado bueno para ser verdad, probablemente lo sea",
      "Pregunte a un familiar o amigo si no está seguro",
    ],
  },
};

// Spam indicator detection patterns (based on actual dataset features)
const spamPatterns = {
  urgency: /\b(urgent|immediately|now|fast|quick|hurry|asap|expire|limited time|act now|don't wait)\b/i,
  freePrize: /\b(free|won|winner|prize|award|gift|congratulations|selected|chosen|lucky)\b/i,
  allCaps: /[A-Z]{4,}/,
  clickLink: /\b(click|tap|link|visit|go to|http|www|\.com|\.net)\b/i,
  personalInfo: /\b(password|pin|ssn|social security|account|verify|confirm|update your)\b/i,
  moneyRequest: /\b(cash|money|\$|dollar|pound|£|credit|bank|pay|payment|refund|claim)\b/i,
  suspiciousNumbers: /\b(\d{4,}|call \d|text \d|code:?\s*\d)\b/i,
  actionWords: /\b(reply|respond|call|text|send|contact|claim|redeem)\b/i,
};

// Analyze text for spam indicators
const detectSpamIndicators = (text) => {
  const detected = [];
  for (const [key, pattern] of Object.entries(spamPatterns)) {
    if (pattern.test(text)) {
      detected.push(key);
    }
  }
  return detected;
};

// Fallback model options if API doesn't respond
const fallbackModels = [
  { id: "logreg", name: "Logistic Regression", description: "Fast, reliable linear classifier" }
];

// Custom styles for react-select dropdown
const customSelectStyles = {
  control: (provided, state) => ({
    ...provided,
    padding: '4px 10px',
    fontFamily: '"Nunito", sans-serif',
    fontSize: '1rem',
    fontWeight: 600,
    color: '#2d3748',
    background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
    border: state.isFocused ? '2.5px solid #2a9d8f' : '2.5px solid #2a9d8f',
    borderRadius: '12px',
    cursor: 'pointer',
    minWidth: '220px',
    boxShadow: state.isFocused 
      ? '0 0 0 4px rgba(42, 157, 143, 0.15), 0 4px 20px rgba(42, 157, 143, 0.3)'
      : '0 2px 8px rgba(42, 157, 143, 0.1)',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    '&:hover': {
      borderColor: '#e76f51',
      background: 'linear-gradient(135deg, #ffffff 0%, #f0f9f8 100%)',
      boxShadow: '0 4px 16px rgba(42, 157, 143, 0.2)',
      transform: 'translateY(-1px)',
    },
  }),
  menu: (provided) => ({
    ...provided,
    borderRadius: '12px',
    overflow: 'hidden',
    boxShadow: '0 8px 24px rgba(42, 157, 143, 0.25)',
    border: '2px solid #2a9d8f',
    marginTop: '8px',
    zIndex: 9999,
  }),
  menuList: (provided) => ({
    ...provided,
    padding: '8px',
    background: '#ffffff',
  }),
  option: (provided, state) => ({
    ...provided,
    fontFamily: '"Nunito", sans-serif',
    fontSize: '1rem',
    fontWeight: 500,
    padding: '12px 16px',
    borderRadius: '8px',
    cursor: 'pointer',
    backgroundColor: state.isSelected 
      ? '#2a9d8f' 
      : state.isFocused 
        ? 'rgba(42, 157, 143, 0.1)' 
        : 'transparent',
    color: state.isSelected ? '#ffffff' : '#2d3748',
    transition: 'all 0.2s ease',
    '&:active': {
      backgroundColor: '#2a9d8f',
    },
  }),
  singleValue: (provided) => ({
    ...provided,
    color: '#2d3748',
    fontWeight: 600,
  }),
  dropdownIndicator: (provided, state) => ({
    ...provided,
    color: '#2a9d8f',
    transition: 'transform 0.3s ease',
    transform: state.selectProps.menuIsOpen ? 'rotate(180deg)' : 'rotate(0deg)',
    '&:hover': {
      color: '#e76f51',
    },
  }),
  indicatorSeparator: () => ({
    display: 'none',
  }),
  menuPortal: (base) => ({
    ...base,
    zIndex: 99999,
  }),
};

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
  
  // Digital literacy state
  const [detectedIndicators, setDetectedIndicators] = useState([]);
  const [showModal, setShowModal] = useState(false);
  
  // Animation state for textarea
  const [textareaAnimation, setTextareaAnimation] = useState("");

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
    setDetectedIndicators([]);
    setTextareaAnimation("");

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: selectedModel }),
      });

      const data = await response.json();
      setPrediction(data.prediction);
      setModelUsed(data.model_used || selectedModel);
      
      // Trigger textarea animation based on result
      if (data.prediction === "1") {
        setTextareaAnimation("spam-pulse");
        const indicators = detectSpamIndicators(text);
        setDetectedIndicators(indicators);
      } else if (data.prediction === "0") {
        setTextareaAnimation("safe-pulse");
      }
      
      // Clear animation after it completes
      setTimeout(() => setTextareaAnimation(""), 1500);
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
    setDetectedIndicators([]);
    setTextareaAnimation("");
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
      
      {/* Help Button - Fixed to page corner */}
      {loaderDone && (
        <button 
          className="help-button" 
          onClick={() => setShowModal(true)}
          aria-label={language === "en" ? "Learn about spam" : "Aprende sobre spam"}
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>
          </svg>
        </button>
      )}
      
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
            <Select
              value={availableModels.find(m => m.id === selectedModel)}
              onChange={(option) => setSelectedModel(option.id)}
              options={availableModels}
              getOptionLabel={(option) => option.name}
              getOptionValue={(option) => option.id}
              styles={customSelectStyles}
              isSearchable={false}
              isDisabled={isLoading}
              className="react-select-container"
              classNamePrefix="react-select"
              menuPortalTarget={document.body}
              menuPosition="fixed"
            />
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
          className={`message-input ${textareaAnimation}`}
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
          const isSpam = prediction === "1";
          return (
            <div className={`${display.className} ${isSpam ? "glow-pulse" : ""}`}>
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

        {/* Learn More Button - Shows when spam detected */}
        {prediction === "1" && (
          <button 
            className="learn-more-btn glow-pulse-btn"
            onClick={() => setShowModal(true)}
          >
            <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>
            </svg>
            {t.learnMoreTitle}
          </button>
        )}
      </div>

      {/* Education Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="modal-header">
              <h2 className="modal-title">{t.learnMoreTitle}</h2>
              <button className="modal-close" onClick={() => setShowModal(false)}>
                <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
              </button>
            </div>

            {/* Warning Signs Detected (if any) */}
            {detectedIndicators.length > 0 && (
              <div className="modal-section warning-signs">
                <h3 className="section-title">{t.whySpamTitle}</h3>
                <div className="indicators-list">
                  {detectedIndicators.map((indicator) => (
                    <div key={indicator} className="indicator-item">
                      <span className="indicator-icon">
                        <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
                          <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
                        </svg>
                      </span>
                      <div className="indicator-content">
                        <span className="indicator-label">{t.indicators[indicator]?.label}</span>
                        <span className="indicator-tip">{t.indicators[indicator]?.tip}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Safety Tips */}
            <div className="modal-section safety-tips">
              <h3 className="section-title">{t.tipsTitle}</h3>
              <ul className="tips-list">
                {t.safetyTips.map((tip, index) => (
                  <li key={index} className="tip-item">
                    <span className="tip-icon">
                      <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                        <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z"/>
                      </svg>
                    </span>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
    </>
  );
}

export default App;
