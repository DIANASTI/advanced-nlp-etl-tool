
#!/usr/bin/env python3

"""
Enhanced ETL tool with NLP metadata extraction (without spaCy dependency).
Extracts keywords, sentiment, and language detection from text files.
"""

import click
import sqlite3
import os
from os import path
from textblob import TextBlob
from langdetect import detect
import yake
from collections import Counter
import json
from datetime import datetime
import re

DATABASE = "enhanced_keywords.db"

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
    # YAKE returns (score, keyword) tuples - score is already float, keyword is string
    return [(kw[1], kw[0]) for kw in keywords]  # Return as (keyword, score)

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

def extract_entities_simple(text):
    """Extract simple entities using regex patterns (fallback for spaCy)"""
    entities = []
    
    # Simple patterns for common entity types
    patterns = {
        'PERSON': r'\b(?:Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        'ORG': r'\b(?:Google|Microsoft|Amazon|Apple|Facebook|Tesla|Netflix|IBM|Oracle|Intel|Adobe|Salesforce|Twitter|LinkedIn|Uber|Airbnb|SpaceX|OpenAI|NVIDIA|AMD|Qualcomm|Cisco|VMware|Zoom|Slack|Dropbox|GitHub|Reddit|YouTube|Instagram|WhatsApp|TikTok|Snapchat|Pinterest|PayPal|Stripe|Square|Robinhood|Coinbase|Binance|Ethereum|Bitcoin)\b',
        'GPE': r'\b(?:San Francisco|New York|Los Angeles|Chicago|Boston|Seattle|Austin|Denver|Miami|Atlanta|Washington|London|Paris|Berlin|Tokyo|Beijing|Shanghai|Mumbai|Delhi|Sydney|Toronto|Vancouver|Montreal)\b',
        'MONEY': r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
        'PERCENT': r'\d+(?:\.\d+)?%',
        'DATE': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
    }
    
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                'text': match,
                'label': label,
                'description': f'{label} entity'
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
    """Create hashtags from keywords"""
    hashtags = []
    
    for item in keywords:
        if isinstance(item, tuple):
            keyword = item[0]
        else:
            keyword = str(item)
        
        if isinstance(keyword, str) and keyword.strip():
            hashtag = '#' + ''.join(word.capitalize() for word in keyword.split() if word.isalpha())
            if len(hashtag) > 1:
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
    entities = extract_entities_simple(text)  # Using simple regex-based extraction
    language = detect_language(text)
    stats = extract_text_stats(text)
    hashtags = create_hashtags(keywords)
    
    return {
        'filename': filename,
        'keywords': keywords,
        'sentiment': sentiment,
        'entities': entities,
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
    
    conn.commit()
    conn.close()
    
    return db_exists

# CLI Commands
@click.group()
def cli():
    """Enhanced NLP ETL Tool (No spaCy version)"""

@cli.command("analyze")
@click.argument("filename", default="sample.txt")
def analyze(filename):
    """Analyze text file with NLP features"""
    click.echo(click.style(f"Analyzing {filename} with NLP features...", fg="green"))
    
    data = load_enhanced_data(filename)
    if not data:
        return
    
    # Display results
    click.echo(click.style(f"\n📊 Analysis Results for {filename}", fg="blue", bold=True))
    
    # Basic stats
    stats = data['stats']
    click.echo(f"\n📈 Text Statistics:")
    click.echo(f"  • Words: {stats['word_count']}")
    click.echo(f"  • Sentences: {stats['sentence_count']}")
    click.echo(f"  • Characters: {stats['char_count']}")
    click.echo(f"  • Avg word length: {stats['avg_word_length']}")
    click.echo(f"  • Language: {data['language']}")
    
    # Sentiment
    sentiment = data['sentiment']
    click.echo(f"\n😊 Sentiment Analysis:")
    click.echo(f"  • Label: {sentiment['label']}")
    click.echo(f"  • Polarity: {sentiment['polarity']} (-1=negative, +1=positive)")
    click.echo(f"  • Subjectivity: {sentiment['subjectivity']} (0=objective, 1=subjective)")
    
    # Keywords
    click.echo(f"\n🔑 Top Keywords:")
    for i, (keyword, score) in enumerate(data['keywords'][:10], 1):
        click.echo(f"  {i:2d}. {keyword} (score: {score:.3f})")
    
    # Entities
    if data['entities']['entities']:
        click.echo(f"\n🏷️  Named Entities:")
        for entity in data['entities']['entities'][:10]:
            click.echo(f"  • {entity['text']} ({entity['label']})")
    
    # Hashtags
    if data['hashtags']:
        click.echo(f"\n#️⃣ Generated Hashtags:")
        hashtag_line = " ".join(data['hashtags'][:10])
        click.echo(f"  {hashtag_line}")
    
    # Save to database
    db_existed = save_to_database(data)
    if db_existed:
        click.echo(click.style(f"\n💾 Data updated in database", fg="green"))
    else:
        click.echo(click.style(f"\n💾 Database created and data saved", fg="green"))

@cli.command("keywords")
@click.option("--limit", default=10, help="Number of keywords to show")
@click.option("--order-by", default="score", help="Order by: score, keyword, filename")
def keywords(limit, order_by):
    """Show extracted keywords from database"""
    if not path.exists(DATABASE):
        click.echo(click.style("No database found. Run 'analyze' first.", fg="red"))
        return
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    if order_by == "score":
        c.execute(f"SELECT keyword, score, filename FROM keywords ORDER BY score ASC LIMIT {limit}")
    else:
        c.execute(f"SELECT keyword, score, filename FROM keywords ORDER BY {order_by} LIMIT {limit}")
    
    results = c.fetchall()
    conn.close()
    
    if results:
        click.echo(click.style(f"\n🔑 Top {len(results)} Keywords:", fg="blue", bold=True))
        for i, (keyword, score, filename) in enumerate(results, 1):
            click.echo(f"  {i:2d}. {keyword} (score: {score:.3f}) - {filename}")
    else:
        click.echo(click.style("No keywords found in database.", fg="yellow"))

@cli.command("sentiment")
@click.option("--filename", help="Filter by specific filename")
def sentiment_cmd(filename):
    """Show sentiment analysis results"""
    if not path.exists(DATABASE):
        click.echo(click.style("No database found. Run 'analyze' first.", fg="red"))
        return
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    if filename:
        c.execute("SELECT filename, polarity, subjectivity, label FROM sentiment WHERE filename = ?", (filename,))
    else:
        c.execute("SELECT filename, polarity, subjectivity, label FROM sentiment")
    
    results = c.fetchall()
    conn.close()
    
    if results:
        click.echo(click.style(f"\n😊 Sentiment Analysis Results:", fg="blue", bold=True))
        for fname, polarity, subjectivity, label in results:
            click.echo(f"  📄 {fname}")
            click.echo(f"     • Label: {label}")
            click.echo(f"     • Polarity: {polarity}")
            click.echo(f"     • Subjectivity: {subjectivity}")
    else:
        click.echo(click.style("No sentiment data found.", fg="yellow"))

@cli.command("entities")
@click.option("--filename", help="Filter by specific filename")
@click.option("--entity-type", help="Filter by entity type (PERSON, ORG, GPE, etc.)")
def entities_cmd(filename, entity_type):
    """Show extracted named entities"""
    if not path.exists(DATABASE):
        click.echo(click.style("No database found. Run 'analyze' first.", fg="red"))
        return
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    query = "SELECT filename, entity_text, entity_label, entity_description FROM entities"
    params = []
    
    if filename and entity_type:
        query += " WHERE filename = ? AND entity_label = ?"
        params = [filename, entity_type]
    elif filename:
        query += " WHERE filename = ?"
        params = [filename]
    elif entity_type:
        query += " WHERE entity_label = ?"
        params = [entity_type]
    
    c.execute(query, params)
    results = c.fetchall()
    conn.close()
    
    if results:
        click.echo(click.style(f"\n🏷️  Named Entities:", fg="blue", bold=True))
        for fname, entity_text, entity_label, entity_desc in results:
            click.echo(f"  📄 {fname}")
            click.echo(f"     • {entity_text} ({entity_label})")
    else:
        click.echo(click.style("No entities found.", fg="yellow"))

@cli.command("stats")
@click.option("--filename", help="Filter by specific filename")
def stats_cmd(filename):
    """Show text statistics"""
    if not path.exists(DATABASE):
        click.echo(click.style("No database found. Run 'analyze' first.", fg="red"))
        return
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    if filename:
        c.execute("SELECT * FROM text_stats WHERE filename = ?", (filename,))
    else:
        c.execute("SELECT * FROM text_stats")
    
    results = c.fetchall()
    conn.close()
    
    if results:
        click.echo(click.style(f"\n📈 Text Statistics:", fg="blue", bold=True))
        for row in results:
            _, fname, word_count, sentence_count, char_count, avg_word_length, language, processed_at = row
            click.echo(f"  📄 {fname}")
            click.echo(f"     • Words: {word_count}")
            click.echo(f"     • Sentences: {sentence_count}")
            click.echo(f"     • Characters: {char_count}")
            click.echo(f"     • Avg word length: {avg_word_length}")
            click.echo(f"     • Language: {language}")
            click.echo(f"     • Processed: {processed_at}")
    else:
        click.echo(click.style("No statistics found.", fg="yellow"))

if __name__ == "__main__":
    cli()
