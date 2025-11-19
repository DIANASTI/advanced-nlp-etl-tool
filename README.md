# Advanced NLP ETL Tool

An enhanced ETL (Extract, Transform, Load) tool for comprehensive text analysis using multiple NLP libraries including spaCy, TextBlob, YAKE, Transformers, and NLTK.

## 🚀 Features

### Core NLP Analysis
- **Keyword Extraction**: Automatic keyword extraction using YAKE algorithm
- **Sentiment Analysis**: 
  - Basic sentiment analysis using TextBlob
  - Advanced sentiment analysis using RoBERTa transformer model
- **Emotion Detection**: Multi-class emotion classification using DistilRoBERTa
- **Named Entity Recognition**: Person, organization, location detection using spaCy
- **Language Detection**: Automatic language identification
- **Part-of-Speech Tagging**: Grammatical analysis using NLTK
- **Readability Metrics**: Flesch Reading Ease and text complexity analysis

### Extended Features
- **Batch Processing**: Process multiple files at once
- **Multiple Export Formats**: JSON, CSV, TXT
- **SQLite Database**: Persistent storage for all analysis results
- **Hashtag Generation**: Automatic hashtag creation from keywords
- **Advanced Text Statistics**: Comprehensive text analysis

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DIANASTI/advanced-nlp-etl-tool.git
   cd advanced-nlp-etl-tool
