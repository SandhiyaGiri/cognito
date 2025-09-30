#!/usr/bin/env python3
"""
Launcher script for the Medical Whisper Streamlit App
"""

import subprocess
import sys
import os
from dotenv import load_dotenv, find_dotenv

def main():
    """Launch the Streamlit app"""
    print("🏥 Starting Medical Whisper Live STT App...")
    print("=" * 50)
    # Load .env from project if present
    try:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
            print(f"🔐 Loaded environment from: {env_path}")
    except Exception:
        pass
    
    # Ensure required dependencies are installed
    required_packages = [
        ("streamlit", "streamlit"),
        ("nltk", "nltk"),
        ("jiwer", "jiwer"),
        ("sklearn", "scikit-learn"),
        ("google.generativeai", "google-generativeai"),
        ("dotenv", "python-dotenv"),
    ]
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name} is available")
        except ImportError:
            print(f"❌ {package_name} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully")
    
    # Launch the app
    print("🚀 Launching Streamlit app...")
    print("📱 The app will open in your browser")
    print("🛑 Press Ctrl+C to stop the app")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped. Goodbye!")

if __name__ == "__main__":
    main()
