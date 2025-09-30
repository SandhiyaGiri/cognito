# 🏥 Medical Whisper Live Speech-to-Text

A comprehensive Streamlit application for real-time medical speech transcription using multiple Whisper models with advanced post-processing and metrics evaluation.

## ✨ Features

### 🎙️ **Multi-Model Transcription**
- **Local Models**: Na0s Medical-Whisper-Large-v3, OpenAI Whisper Large v3
- **Cloud Models**: Groq Whisper Large v3 (API), Groq Whisper Large v3 Turbo (API)
- **Real-time Recording**: Live audio capture with streamlit-audiorecorder
- **File Upload**: Support for WAV, MP3, M4A, OGG, FLAC formats

### 🧠 **Advanced Post-Processing**
- **Gemini LLM Integration**: AI-powered transcription cleaning and correction
- **Medical Terminology**: Specialized prompts for medical transcription accuracy
- **Conservative Processing**: Maintains original meaning while fixing errors

### 📊 **Comprehensive Metrics**
- **WER/CER Calculation**: Word and Character Error Rates with NLTK stopword removal
- **Cosine Similarity**: Semantic similarity between ground truth and transcriptions
- **Model Comparison**: Side-by-side performance analysis
- **Processing Times**: Real-time performance metrics

### 📈 **Data Management**
- **Recordings Table**: Persistent storage of all transcriptions
- **CSV Export**: Downloadable results with metrics
- **Session Management**: Maintains state across interactions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster local inference)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd medical-whisper-cognito
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   # Required for Groq API models
   GROQ_API_KEY=your_groq_api_key_here
   
   # Required for Gemini post-processing
   GOOGLE_API_KEY=your_google_api_key_here
   
   # Optional: Specify Gemini model (default: gemini-2.0-flash-lite)
   DEFAULT_MODEL=gemini-2.0-flash-lite
   ```

4. **Launch the application**
   ```bash
   python run_app.py
   ```

The app will automatically:
- Install missing dependencies
- Load environment variables
- Start the Streamlit server on `http://localhost:8501`

## 📋 Usage Guide

### 1. **Model Configuration**
- Select models from the sidebar (default: all except Crystalcareai)
- Choose device (auto, CPU, CUDA, MPS)
- Click "Load Selected Models"

### 2. **Ground Truth & Post-Processing**
- Enter ground truth text (optional) for metrics calculation
- Enable Gemini post-processing for enhanced accuracy
- Metrics are computed after NLTK stopword removal

### 3. **Recording & Transcription**
- **Live Recording**: Click microphone button to start/stop
- **File Upload**: Upload audio files for batch processing
- **Model Comparison**: View results from all selected models

### 4. **Results & Export**
- **Comparison Table**: Shows raw and processed outputs with metrics
- **Recordings Table**: Persistent storage of all sessions
- **CSV Download**: Export results with full metrics

## 🔧 Configuration

### Model Selection
```python
model_options = {
    "Na0s/Medical-Whisper-Large-v3": "Medical-optimized Whisper",
    "openai/whisper-large-v3": "Standard Whisper Large v3",
    "groq/whisper-large-v3": "Groq API Whisper",
    "groq/whisper-large-v3-turbo": "Groq API Whisper Turbo"
}
```

### Metrics Configuration
- **Stopword Removal**: Uses NLTK English stopwords
- **WER/CER**: Computed with jiwer library
- **Cosine Similarity**: TF-IDF vectorization with scikit-learn

## 📊 Metrics Explained

### **Word Error Rate (WER)**
- Measures word-level accuracy: `(S + D + I) / N`
- S = substitutions, D = deletions, I = insertions, N = total words
- Lower is better (0% = perfect)

### **Character Error Rate (CER)**
- Measures character-level accuracy
- Useful for spelling and punctuation errors
- Lower is better (0% = perfect)

### **Cosine Similarity**
- Measures semantic similarity between texts
- Range: 0 to 1 (1 = identical meaning)
- Higher is better

## 🛠️ Technical Details

### **Architecture**
- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python with transformers, librosa, soundfile
- **APIs**: Groq API for cloud inference
- **LLM**: Google Gemini for post-processing

### **Audio Processing**
- **Sample Rate**: 16kHz (Whisper standard)
- **Format**: WAV conversion for compatibility
- **Chunking**: 30s chunks with 5s stride for long audio

### **Performance Optimization**
- **Device Selection**: Auto-detection of CUDA/MPS/CPU
- **Model Caching**: Persistent model loading
- **Batch Processing**: Efficient multi-model inference

## 📁 Project Structure

```
cognito/
├── streamlit_app.py          # Main Streamlit application
├── run_app.py               # Application launcher
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables (create this)
```

## 🔑 API Keys Setup

### **Groq API**
1. Visit [Groq Console](https://console.groq.com/)
2. Create account and generate API key
3. Add to `.env`: `GROQ_API_KEY=your_key`

### **Google Gemini API**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create API key
3. Add to `.env`: `GOOGLE_API_KEY=your_key`

## 🐛 Troubleshooting

### **Common Issues**

1. **"streamlit-audiorecorder not available"**
   ```bash
   pip install streamlit-audiorecorder
   ```

2. **"NLTK stopwords not found"**
   - App automatically downloads stopwords
   - Manual: `python -c "import nltk; nltk.download('stopwords')"`

3. **"CUDA out of memory"**
   - Use CPU mode in device selection
   - Reduce batch size or use smaller models

4. **"API key not found"**
   - Check `.env` file exists and contains correct keys
   - Verify environment variables are loaded

### **Performance Tips**
- Use Groq API models for fastest inference
- Enable GPU acceleration for local models
- Use Gemini post-processing for best accuracy

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper**: Base speech recognition models
- **Na0s**: Medical-optimized Whisper fine-tuning
- **Groq**: Fast cloud inference platform
- **Google Gemini**: LLM post-processing
- **Streamlit**: Web application framework

## 📞 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the Streamlit documentation

---

**Made with ❤️ for medical transcription accuracy**
