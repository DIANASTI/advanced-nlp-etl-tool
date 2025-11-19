
#!/usr/bin/env python3

"""
Enhanced ETL tool with advanced NLP metadata extraction.
Extracts keywords, sentiment, entities, and language detection from text files.
"""

import click
import sqlite3
import os
from os import path
import spacy
from textblob import TextBlob
from langdetect import detect
import yake
from collections import Counter
import json
from datetime import datetime

# Add these imports after the existing ones
import warnings
warnings.filterwarnings("ignore")

# New imports for extended functionality
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Install with: pip install nltk")

DATABASE = "enhanced_keywords.db"

# Load spaCy model (install with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

# Add after the nlp model loading
# Initialize transformers pipelines
emotion_classifier = None
advanced_sentiment = None

if TRANSFORMERS_AVAILABLE:
    try:
        # Emotion classification
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Advanced sentiment analysis
        advanced_sentiment = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        print("✅ Transformers models loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load some transformers models: {e}")

def read_file(filename):
    """Read text from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        click.echo(click.style(f"File {filename} not found", fg="red"))
        return ""

def extract_keywords_yake(text, max_keywords=20):
    """Extract keywords using YAKE algorithm"""
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,  # n-gram size
        dedupLim=0.7,
        top=max_keywords
    )
    keywords = kw_extractor.extract_keywords(text)
    return [(kw[1], kw[0]) for kw in keywords]  # (keyword, score)

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Classify sentiment
    if sentiment.polarity > 0.1:
        sentiment_label = "positive"
    elif sentiment.polarity < -0.1:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    
    return {
        'polarity': round(sentiment.polarity, 3),
        'subjectivity': round(sentiment.subjectivity, 3),
        'label': sentiment_label
    }

def extract_entities(text):
    """Extract named entities using spaCy"""
    if not nlp:
        return {
            'entities': [],
            'entity_counts': {},
            'total_entities': 0
        }
    
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'description': spacy.explain(ent.label_)
        })
    
    # Count entity types
    entity_counts = Counter([ent['label'] for ent in entities])
    
    return {
        'entities': entities[:10],  # Limit to top 10
        'entity_counts': dict(entity_counts),
        'total_entities': len(entities)
    }

def detect_language(text):
    """Detect language of the text"""
    try:
        return detect(text)
    except:
        return "unknown"

def extract_text_stats(text):
    """Extract basic text statistics"""
    words = text.split()
    sentences = text.split('.')
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'char_count': len(text),
        'avg_word_length': round(sum(len(word) for word in words) / len(words), 2) if words else 0
    }

def create_hashtags(keywords):
    """Create hashtags from keywords (expects list of tuples or strings)"""
    hashtags = []
    
    for item in keywords:
        # Handle both tuple format (keyword, score) and string format
        if isinstance(item, tuple):
            keyword = item[0]  # Extract keyword from tuple
        else:
            keyword = str(item)  # Convert to string if not tuple
        
        # Clean and format the keyword
        if isinstance(keyword, str) and keyword.strip():
            # Remove special characters and create hashtag
            hashtag = '#' + ''.join(word.capitalize() for word in keyword.split() if word.isalpha())
            if len(hashtag) > 1:  # Only add if hashtag has content after #
                hashtags.append(hashtag)
    
    return hashtags

def load_enhanced_data(filename):
    """Load and analyze text with enhanced NLP features"""
    text = read_file(filename)
    if not text:
        return None
    
    # Extract all NLP features
    keywords = extract_keywords_yake(text)
    sentiment = analyze_sentiment(text)
    advanced_sentiment_result = analyze_advanced_sentiment(text)
    emotions = analyze_emotions(text)
    entities = extract_entities(text)
    pos_tags = extract_pos_tags(text)
    readability = extract_readability_metrics(text)
    language = detect_language(text)
    stats = extract_text_stats(text)
    hashtags = create_hashtags(keywords)
    
    return {
        'filename': filename,
        'keywords': keywords,
        'sentiment': sentiment,
        'advanced_sentiment': advanced_sentiment_result,
        'emotions': emotions,
        'entities': entities,
        'pos_tags': pos_tags,
        'readability': readability,
        'language': language,
        'stats': stats,
        'hashtags': hashtags,
        'processed_at': datetime.now().isoformat()
    }

def create_enhanced_database():
    """Create enhanced database schema"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Main keywords table
    c.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            keyword TEXT,
            score REAL,
            processed_at TEXT
        )
    ''')
    
    # Sentiment analysis table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            polarity REAL,
            subjectivity REAL,
            label TEXT,
            processed_at TEXT
        )
    ''')
    
    # Advanced sentiment table
    c.execute('''
        CREATE TABLE IF NOT EXISTS advanced_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label TEXT,
            confidence REAL,
            processed_at TEXT
        )
    ''')
    
    # Emotions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            top_emotion TEXT,
            top_emotion_score REAL,
            all_emotions TEXT,
            processed_at TEXT
        )
    ''')
    
    # Named entities table
    c.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            entity_text TEXT,
            entity_label TEXT,
            entity_description TEXT,
            processed_at TEXT
        )
    ''')
    
    # POS tags table
    c.execute('''
        CREATE TABLE IF NOT EXISTS pos_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            pos_counts TEXT,
            total_words INTEGER,
            processed_at TEXT
        )
    ''')
    
    # Readability table
    c.execute('''
        CREATE TABLE IF NOT EXISTS readability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            flesch_reading_ease REAL,
            avg_sentence_length REAL,
            avg_syllables_per_word REAL,
            processed_at TEXT
        )
    ''')
    
    # Text statistics table
    c.execute('''
        CREATE TABLE IF NOT EXISTS text_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            word_count INTEGER,
            sentence_count INTEGER,
            char_count INTEGER,
            avg_word_length REAL,
            language TEXT,
            processed_at TEXT
        )
    ''')
    
    # Hashtags table
    c.execute('''
        CREATE TABLE IF NOT EXISTS hashtags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            hashtag TEXT,
            processed_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_to_database(data):
    """Save enhanced data to database"""
    if not data:
        return False
    
    db_exists = path.exists(DATABASE)
    create_enhanced_database()
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    filename = data['filename']
    processed_at = data['processed_at']
    
    # Save keywords
    for keyword, score in data['keywords']:
        c.execute(
            "INSERT INTO keywords (filename, keyword, score, processed_at) VALUES (?, ?, ?, ?)",
            (filename, keyword, score, processed_at)
        )
    
    # Save sentiment
    sentiment = data['sentiment']
    c.execute(
        "INSERT INTO sentiment (filename, polarity, subjectivity, label, processed_at) VALUES (?, ?, ?, ?, ?)",
        (filename, sentiment['polarity'], sentiment['subjectivity'], sentiment['label'], processed_at)
    )
    
    # Save advanced sentiment
    if data['advanced_sentiment']:
        adv_sent = data['advanced_sentiment']
        c.execute(
            "INSERT INTO advanced_sentiment (filename, label, confidence, processed_at) VALUES (?, ?, ?, ?)",
            (filename, adv_sent['label'], adv_sent['confidence'], processed_at)
        )
    
    # Save emotions
    if data['emotions']:
        emotions = data['emotions']
        c.execute(
            "INSERT INTO emotions (filename, top_emotion, top_emotion_score, all_emotions, processed_at) VALUES (?, ?, ?, ?, ?)",
            (filename, emotions['top_emotion'], emotions['top_emotion_score'], 
             json.dumps(emotions['all_emotions']), processed_at)
        )
    
    # Save entities
    for entity in data['entities']['entities']:
        c.execute(
            "INSERT INTO entities (filename, entity_text, entity_label, entity_description, processed_at) VALUES (?, ?, ?, ?, ?)",
            (filename, entity['text'], entity['label'], entity['description'], processed_at)
        )
    
    # Save text stats
    stats = data['stats']
    c.execute(
        "INSERT INTO text_stats (filename, word_count, sentence_count, char_count, avg_word_length, language, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (filename, stats['word_count'], stats['sentence_count'], stats['char_count'], stats['avg_word_length'], data['language'], processed_at)
    )
    
    # Save hashtags
    for hashtag in data['hashtags']:
        c.execute(
            "INSERT INTO hashtags (filename, hashtag, processed_at) VALUES (?, ?, ?)",
            (filename, hashtag, processed_at)
        )
    
    # Save POS tags
    pos_data = data['pos_tags']
    c.execute(
        "INSERT INTO pos_tags (filename, pos_counts, total_words, processed_at) VALUES (?, ?, ?, ?)",
        (filename, json.dumps(pos_data['pos_counts']), pos_data['total_words'], processed_at)
    )
    
    # Save readability metrics
    if data['readability']:
        readability = data['readability']
        c.execute(
            "INSERT INTO readability (filename, flesch_reading_ease, avg_sentence_length, avg_syllables_per_word, processed_at) VALUES (?, ?, ?, ?, ?)",
            (filename, 
             readability.get('flesch_reading_ease'),
             readability.get('avg_sentence_length'),
             readability.get('avg_syllables_per_word'),
             processed_at)
        )
    
    conn.commit()
    conn.close()
    
    return db_exists

def query_keywords(limit=10, order_by="score"):
    """Query keywords from database"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    if order_by == "score":
        c.execute(f"SELECT keyword, score, filename FROM keywords ORDER BY score ASC LIMIT {limit}")
    else:
        c.execute(f"SELECT keyword, score, filename FROM keywords ORDER BY {order_by} LIMIT {limit}")
    
    results = c.fetchall()
    conn.close()
    return results

def query_sentiment(filename=None):
    """Query sentiment analysis results"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    if filename:
        c.execute("SELECT * FROM sentiment WHERE filename = ?", (filename,))
    else:
        c.execute("SELECT * FROM sentiment")
    
    results = c.fetchall()
    conn.close()
    return results

def query_entities(limit=20):
    """Query named entities"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(f"SELECT entity_text, entity_label, entity_description, filename FROM entities LIMIT {limit}")
    results = c.fetchall()
    conn.close()
    return results

def query_stats():
    """Query text statistics"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT * FROM text_stats")
    results = c.fetchall()
    conn.close()
    return results

def analyze_advanced_sentiment(text):
    """Analyze sentiment using advanced transformer model"""
    if not TRANSFORMERS_AVAILABLE or not advanced_sentiment or not text.strip():
        return None
    
    try:
        # Truncate text if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = advanced_sentiment(text)[0]
        return {
            'label': result['label'].lower(),
            'confidence': round(result['score'], 3)
        }
    except Exception as e:
        print(f"Error in advanced sentiment analysis: {e}")
        return None

def analyze_emotions(text):
    """Analyze emotions using transformer model"""
    if not TRANSFORMERS_AVAILABLE or not emotion_classifier or not text.strip():
        return None
    
    try:
        # Truncate text if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        emotions = emotion_classifier(text)[0]
        
        # Get top 3 emotions
        emotions_sorted = sorted(emotions, key=lambda x: x['score'], reverse=True)[:3]
        
        return {
            'top_emotion': emotions_sorted[0]['label'],
            'top_emotion_score': round(emotions_sorted[0]['score'], 3),
            'all_emotions': {emotion['label']: round(emotion['score'], 3) for emotion in emotions}
        }
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return None

def extract_pos_tags(text):
    """Extract Part-of-Speech tags using NLTK"""
    if not NLTK_AVAILABLE or not text.strip():
        return {
            'pos_counts': {},
            'total_words': 0
        }
    
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Count POS tags
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        return {
            'pos_counts': dict(pos_counts),
            'total_words': len(tokens)
        }
    except Exception as e:
        print(f"Error in POS tagging: {e}")
        return {
            'pos_counts': {},
            'total_words': 0
        }

def extract_readability_metrics(text):
    """Calculate readability metrics"""
    if not text.strip():
        return {}
    
    try:
        # Calculate basic readability metrics manually
        words = text.split()
        sentences = text.split('.')
        syllables = sum([len([c for c in word if c.lower() in 'aeiou']) for word in words])
        
        if len(sentences) > 0 and len(words) > 0:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            # Simple Flesch Reading Ease approximation
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            return {
                'flesch_reading_ease': round(flesch_score, 2),
                'avg_sentence_length': round(avg_sentence_length, 2),
                'avg_syllables_per_word': round(avg_syllables_per_word, 2)
            }
        
        return {}
    except Exception as e:
        print(f"Error calculating readability: {e}")
        return {}

# CLI Commands
@click.group()
def cli():
    """Enhanced NLP ETL Tool with advanced text analysis"""

@cli.command("analyze")
@click.argument("filename", default="sample.txt")
def analyze(filename):
    """Analyze text file with enhanced NLP features
    
    Example:
    python enhanced_etl.py analyze sample.txt
    """
    click.echo(click.style(f"Analyzing {filename} with enhanced NLP features...", fg="green"))
    
    data = load_enhanced_data(filename)
    if not data:
        return
    
    db_existed = save_to_database(data)
    
    # Display enhanced summary
    click.echo(click.style(f"\n📊 Enhanced Analysis Summary for {filename}:", fg="cyan", bold=True))
    click.echo(f"Language: {data['language']}")
    click.echo(f"Words: {data['stats']['word_count']}")
    click.echo(f"Sentences: {data['stats']['sentence_count']}")
    click.echo(f"Basic Sentiment: {data['sentiment']['label']} (polarity: {data['sentiment']['polarity']})")
    
    if data['advanced_sentiment']:
        click.echo(f"Advanced Sentiment: {data['advanced_sentiment']['label']} (confidence: {data['advanced_sentiment']['confidence']})")
    
    if data['emotions']:
        click.echo(f"Top Emotion: {data['emotions']['top_emotion']} (score: {data['emotions']['top_emotion_score']})")
    
    click.echo(f"Keywords found: {len(data['keywords'])}")
    click.echo(f"Entities found: {data['entities']['total_entities']}")
    click.echo(f"POS tags analyzed: {data['pos_tags']['total_words']} words")
    
    if data['readability']:
        flesch_score = data['readability'].get('flesch_reading_ease')
        if flesch_score:
            click.echo(f"Readability (Flesch): {flesch_score}")
    
    click.echo(f"Hashtags: {', '.join(data['hashtags'][:5])}")
    
    if db_existed:
        click.echo(click.style("Data added to existing database", fg="yellow"))
    else:
        click.echo(click.style("Database created and data saved", fg="green"))

@cli.command("keywords")
@click.option("--limit", default=10, help="Number of keywords to show")
@click.option("--order_by", default="score", help="Order by: score, keyword, filename")
def keywords(limit, order_by):
    """Query keywords from database
    
    Example:
    python enhanced_etl.py keywords --limit 5
    """
    results = query_keywords(limit, order_by)
    
    click.echo(click.style(f"\n🔑 Top {len(results)} Keywords:", fg="cyan", bold=True))
    for keyword, score, filename in results:
        click.echo(f"{click.style(keyword, fg='red')} | {click.style(f'{score:.3f}', fg='green')} | {click.style(filename, fg='blue')}")

@cli.command("sentiment")
@click.option("--filename", help="Filter by filename")
def sentiment(filename):
    """Query sentiment analysis results
    
    Example:
    python enhanced_etl.py sentiment --filename sample.txt
    """
    results = query_sentiment(filename)
    
    click.echo(click.style(f"\n😊 Sentiment Analysis Results:", fg="cyan", bold=True))
    for result in results:
        id_, fname, polarity, subjectivity, label, processed_at = result
        click.echo(f"{click.style(fname, fg='blue')} | {click.style(label.upper(), fg='green')} | Polarity: {polarity} | Subjectivity: {subjectivity}")

@cli.command("entities")
@click.option("--limit", default=20, help="Number of entities to show")
def entities(limit):
    """Query named entities
    
    Example:
    python enhanced_etl.py entities --limit 10
    """
    results = query_entities(limit)
    
    click.echo(click.style(f"\n🏷️  Named Entities:", fg="cyan", bold=True))
    for entity_text, entity_label, entity_description, filename in results:
        click.echo(f"{click.style(entity_text, fg='red')} | {click.style(entity_label, fg='green')} | {click.style(entity_description or 'N/A', fg='yellow')} | {click.style(filename, fg='blue')}")

@cli.command("stats")
def stats():
    """Show text statistics
    
    Example:
    python enhanced_etl.py stats
    """
    results = query_stats()
    
    click.echo(click.style(f"\n📈 Text Statistics:", fg="cyan", bold=True))
    for result in results:
        id_, filename, word_count, sentence_count, char_count, avg_word_length, language, processed_at = result
        click.echo(f"{click.style(filename, fg='blue')} | Words: {word_count} | Sentences: {sentence_count} | Chars: {char_count} | Avg word length: {avg_word_length} | Language: {language}")

@cli.command("delete")
def delete():
    """Delete the database
    
    Example:
    python enhanced_etl.py delete
    """
    if path.exists(DATABASE):
        path_to_db = path.abspath(DATABASE)
        click.echo(click.style(f"Deleting database: {path_to_db}", fg="green"))
        os.remove(DATABASE)
        click.echo(click.style("Database deleted successfully", fg="green"))
    else:
        click.echo(click.style("Database does not exist", fg="red"))

@cli.command("export")
@click.argument("filename", default="sample.txt")
@click.option("--output-format", default="json", type=click.Choice(['json', 'csv', 'txt']), help="Output format")
@click.option("--output-file", help="Output file name (optional)")
def export(filename, output_format, output_file):
    """Export analysis results in different formats
    
    Example:
    python enhanced_etl.py export sample.txt --output-format json
    python enhanced_etl.py export sample.txt --output-format csv --output-file results.csv
    """
    click.echo(click.style(f"Exporting analysis of {filename} in {output_format} format...", fg="green"))
    
    data = load_enhanced_data(filename)
    if not data:
        return
    
    # Generate output filename if not provided
    if not output_file:
        base_name = path.splitext(filename)[0]
        output_file = f"{base_name}_analysis.{output_format}"
    
    try:
        if output_format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif output_format == "csv":
            import csv
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write headers and data
                writer.writerow(['Category', 'Key', 'Value'])
                
                # Basic info
                writer.writerow(['File', 'filename', data['filename']])
                writer.writerow(['File', 'language', data['language']])
                writer.writerow(['File', 'processed_at', data['processed_at']])
                
                # Stats
                for key, value in data['stats'].items():
                    writer.writerow(['Stats', key, value])
                
                # Sentiment
                for key, value in data['sentiment'].items():
                    writer.writerow(['Sentiment', key, value])
                
                # Advanced sentiment
                if data.get('advanced_sentiment'):
                    for key, value in data['advanced_sentiment'].items():
                        writer.writerow(['Advanced_Sentiment', key, value])
                
                # Emotions
                if data.get('emotions'):
                    for key, value in data['emotions'].items():
                        writer.writerow(['Emotions', key, value])
                
                # Keywords
                for keyword, score in data['keywords']:
                    writer.writerow(['Keywords', keyword, score])
                
                # Entities
                for entity in data['entities']['entities']:
                    writer.writerow(['Entities', entity['text'], f"{entity['label']} - {entity['description']}"])
                
                # Hashtags
                for hashtag in data['hashtags']:
                    writer.writerow(['Hashtags', hashtag, ''])
        
        elif output_format == "txt":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Enhanced NLP Analysis Report\n")
                f.write(f"{'='*50}\n\n")
                
                f.write(f"File: {data['filename']}\n")
                f.write(f"Language: {data['language']}\n")
                f.write(f"Processed: {data['processed_at']}\n\n")
                
                f.write(f"Text Statistics:\n")
                f.write(f"- Words: {data['stats']['word_count']}\n")
                f.write(f"- Sentences: {data['stats']['sentence_count']}\n")
                f.write(f"- Characters: {data['stats']['char_count']}\n")
                f.write(f"- Average word length: {data['stats']['avg_word_length']}\n\n")
                
                f.write(f"Sentiment Analysis:\n")
                f.write(f"- Label: {data['sentiment']['label']}\n")
                f.write(f"- Polarity: {data['sentiment']['polarity']}\n")
                f.write(f"- Subjectivity: {data['sentiment']['subjectivity']}\n\n")
                
                if data.get('advanced_sentiment'):
                    f.write(f"Advanced Sentiment Analysis:\n")
                    f.write(f"- Label: {data['advanced_sentiment']['label']}\n")
                    f.write(f"- Confidence: {data['advanced_sentiment']['confidence']}\n\n")
                
                if data.get('emotions'):
                    f.write(f"Emotion Analysis:\n")
                    f.write(f"- Top Emotion: {data['emotions']['top_emotion']}\n")
                    f.write(f"- Top Emotion Score: {data['emotions']['top_emotion_score']}\n\n")
                
                f.write(f"Top Keywords:\n")
                for keyword, score in data['keywords'][:10]:
                    f.write(f"- {keyword} (score: {score:.3f})\n")
                f.write("\n")
                
                f.write(f"Named Entities:\n")
                for entity in data['entities']['entities'][:10]:
                    f.write(f"- {entity['text']} ({entity['label']}: {entity['description']})\n")
                f.write("\n")
                
                f.write(f"Hashtags:\n")
                f.write(f"{', '.join(data['hashtags'])}\n")
        
        click.echo(click.style(f"✅ Analysis exported to {output_file}", fg="green"))
        
    except Exception as e:
        click.echo(click.style(f"❌ Error exporting to {output_format}: {e}", fg="red"))

@cli.command("batch")
@click.argument("directory", default=".")
@click.option("--pattern", default="*.txt", help="File pattern to match")
@click.option("--export-format", default="json", type=click.Choice(['json', 'csv', 'txt']), help="Export format for results")
def batch(directory, pattern, export_format):
    """Batch process multiple files in a directory
    
    Example:
    python enhanced_etl.py batch . --pattern "*.txt" --export-format json
    """
    import glob
    
    # Find files matching pattern
    search_pattern = path.join(directory, pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        click.echo(click.style(f"No files found matching pattern: {search_pattern}", fg="red"))
        return
    
    click.echo(click.style(f"Found {len(files)} files to process...", fg="green"))
    
    batch_results = []
    
    for file_path in files:
        click.echo(f"Processing {file_path}...")
        
        data = load_enhanced_data(file_path)
        if data:
            save_to_database(data)
            batch_results.append(data)
            click.echo(click.style(f"✅ Processed {file_path}", fg="green"))
        else:
            click.echo(click.style(f"❌ Failed to process {file_path}", fg="red"))
    
    # Export batch results
    if batch_results:
        batch_output_file = f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
        
        try:
            if export_format == "json":
                with open(batch_output_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_results, f, indent=2, ensure_ascii=False)
            
            elif export_format == "csv":
                import csv
                with open(batch_output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Filename', 'Language', 'Words', 'Sentences', 'Sentiment', 'Top_Keywords', 'Entity_Count'])
                    
                    for data in batch_results:
                        top_keywords = '; '.join([kw[0] for kw in data['keywords'][:5]])
                        writer.writerow([
                            data['filename'],
                            data['language'],
                            data['stats']['word_count'],
                            data['stats']['sentence_count'],
                            data['sentiment']['label'],
                            top_keywords,
                            data['entities']['total_entities']
                        ])
            
            click.echo(click.style(f"✅ Batch results exported to {batch_output_file}", fg="green"))
            
        except Exception as e:
            click.echo(click.style(f"❌ Error exporting batch results: {e}", fg="red"))
    
    click.echo(click.style(f"\n📊 Batch processing complete! Processed {len(batch_results)} files successfully.", fg="cyan", bold=True))

@cli.command("emotions")
@click.option("--filename", help="Filter by filename")
def emotions(filename):
    """Query emotion analysis results
    
    Example:
    python enhanced_etl.py emotions --filename sample.txt
    """
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    if filename:
        c.execute("SELECT * FROM emotions WHERE filename = ?", (filename,))
    else:
        c.execute("SELECT * FROM emotions ORDER BY top_emotion_score DESC LIMIT 20")
    
    results = c.fetchall()
    conn.close()
    
    click.echo(click.style(f"\n😊 Emotion Analysis Results:", fg="cyan", bold=True))
    for result in results:
        id_, fname, top_emotion, score, all_emotions, processed_at = result
        click.echo(f"{click.style(fname, fg='blue')} | {click.style(top_emotion.upper(), fg='green')} | Score: {score}")

@cli.command("pos")
@click.option("--filename", help="Filter by filename")
def pos(filename):
    """Query Part-of-Speech analysis results
    
    Example:
    python enhanced_etl.py pos --filename sample.txt
    """
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    if filename:
        c.execute("SELECT * FROM pos_tags WHERE filename = ?", (filename,))
    else:
        c.execute("SELECT * FROM pos_tags LIMIT 10")
    
    results = c.fetchall()
    conn.close()
    
    click.echo(click.style(f"\n🏷️ Part-of-Speech Analysis Results:", fg="cyan", bold=True))
    for result in results:
        id_, fname, pos_counts_json, total_words, processed_at = result
        pos_counts = json.loads(pos_counts_json)
        top_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        click.echo(f"{click.style(fname, fg='blue')} | Total words: {total_words}")
        click.echo(f"  Top POS tags: {', '.join([f'{tag}({count})' for tag, count in top_pos])}")

@cli.command("readability")
@click.option("--filename", help="Filter by filename")
def readability(filename):
    """Query readability analysis results
    
    Example:
    python enhanced_etl.py readability --filename sample.txt
    """
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    if filename:
        c.execute("SELECT * FROM readability WHERE filename = ?", (filename,))
    else:
        c.execute("SELECT * FROM readability LIMIT 10")
    
    results = c.fetchall()
    conn.close()
    
    click.echo(click.style(f"\n📖 Readability Analysis Results:", fg="cyan", bold=True))
    for result in results:
        id_, fname, flesch_score, avg_sent_len, avg_syll, processed_at = result
        click.echo(f"{click.style(fname, fg='blue')} | Flesch Score: {flesch_score} | Avg Sentence Length: {avg_sent_len}")

if __name__ == "__main__":
    cli()
