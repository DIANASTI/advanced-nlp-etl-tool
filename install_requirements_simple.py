
#!/usr/bin/env python3

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("🚀 Installing Enhanced NLP ETL Tool Requirements...")
    
    # Install packages
    packages = [
        "click>=8.0.0",
        "spacy>=3.4.0", 
        "textblob>=0.17.0",
        "langdetect>=1.0.9",
        "yake>=0.4.8",
        "nltk>=3.7"
    ]
    
    print("\n📦 Installing Python packages...")
    for package in packages:
        print(f"Installing {package}...")
        success, output = run_command(f"{sys.executable} -m pip install {package}")
        if success:
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}: {output}")
    
    print("\n📥 Downloading models...")
    
    # Download spaCy model
    print("Downloading spaCy English model...")
    success, output = run_command(f"{sys.executable} -m spacy download en_core_web_sm")
    if success:
        print("✅ spaCy model downloaded successfully")
    else:
        print(f"❌ Failed to download spaCy model: {output}")
    
    # Download NLTK data
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('brown', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
    
    print("\n🎉 Installation process completed!")
    print("Run 'python check_dependencies.py' to verify everything is working.")

if __name__ == "__main__":
    main()
