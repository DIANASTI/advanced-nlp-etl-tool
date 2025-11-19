#!/bin/bash

echo "Installing Enhanced NLP ETL Tool..."

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# Download TextBlob corpora
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"

echo "Installation complete!"
