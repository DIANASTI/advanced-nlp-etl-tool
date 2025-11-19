
#!/usr/bin/env python3

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def download_models():
    """Download required models"""
    try:
        # Download spaCy English model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ Successfully downloaded spaCy English model")
        
        # Download TextBlob corpora
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('brown', quiet=True)
        print("✅ Successfully downloaded TextBlob corpora")
        return True
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False

def main():
    print("🚀 Installing Enhanced NLP ETL Tool Requirements...")
    
    # List of required packages
    packages = [
        "click>=8.0.0",
        "spacy>=3.4.0", 
        "textblob>=0.17.0",
        "langdetect>=1.0.9",
        "yake>=0.4.8",
        "nltk>=3.7"
    ]
    
    # Install packages
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
        return False
    
    print("\n📦 All packages installed successfully!")
    
    # Download models
    print("\n📥 Downloading required models...")
    if download_models():
        print("\n🎉 Installation completed successfully!")
        print("\nYou can now run: python enhanced_etl.py --help")
        return True
    else:
        print("\n⚠️  Package installation succeeded but model download failed.")
        print("You may need to run these manually:")
        print("python -m spacy download en_core_web_sm")
        # Fixed the quote issue here
        nltk_command = 'python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"'
        print(nltk_command)
        return False

if __name__ == "__main__":
    main()
