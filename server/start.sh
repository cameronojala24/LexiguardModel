#!/bin/bash
# Start script for Render deployment

# Download NLTK data (needed for Naive Bayes)
python3 -c "import nltk; nltk.download('punkt', quiet=True)" || true

# Start gunicorn on the PORT environment variable that Render provides
exec gunicorn -b 0.0.0.0:${PORT:-8000} app:app
