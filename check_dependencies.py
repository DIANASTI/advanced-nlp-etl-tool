
#!/usr/bin/env python3

def check_module(module_name, import_statement=None):
    """Check if a module can be imported"""
    try:
        if import_statement:
            exec(import_statement)
        else:
            __import__(module_name)
        print(f"✅ {module_name} - Available")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - Missing ({e})")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - Error ({e})")
        return False

def main():
    print("🔍 Checking Enhanced NLP ETL Tool Dependencies...\n")
    
    # Built-in modules (should always be available)
    builtin_modules = [
        "sqlite3", "os", "json", "datetime", "collections"
    ]
    
    # External modules that need installation
    external_modules = [
        ("click", None),
        ("spacy", None),
        ("textblob", None),
        ("langdetect", None),
        ("yake", None),
        ("nltk", None)
    ]
    
    # Special checks
    special_checks = [
        ("spacy_model", "import spacy; nlp = spacy.load('en_core_web_sm')"),
        ("nltk_data", "import nltk; nltk.data.find('tokenizers/punkt')")
    ]
    
    print("📦 Built-in modules:")
    builtin_ok = True
    for module in builtin_modules:
        if not check_module(module):
            builtin_ok = False
    
    print("\n📦 External modules:")
    external_ok = True
    for module_name, import_stmt in external_modules:
        if not check_module(module_name, import_stmt):
            external_ok = False
    
    print("\n📦 Special requirements:")
    special_ok = True
    for check_name, import_stmt in special_checks:
        if not check_module(check_name, import_stmt):
            special_ok = False
    
    print("\n" + "="*50)
    if builtin_ok and external_ok and special_ok:
        print("🎉 All dependencies are satisfied!")
        print("You can run: python enhanced_etl.py --help")
    else:
        print("❌ Some dependencies are missing.")
        if not external_ok:
            print("Run: python install_requirements_simple.py")
        if not special_ok:
            print("Run: python -m spacy download en_core_web_sm")
            # Corrected quote usage for NLTK command
            nltk_cmd = 'python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'brown\')"'
            print(f"Run: {nltk_cmd}")

if __name__ == "__main__":
    main()
