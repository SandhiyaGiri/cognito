#!/usr/bin/env python3
"""
Streamlit App for Medical Whisper Live Speech-to-Text using streamlit-audiorecorder
"""

import streamlit as st
import torch
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
import io
import tempfile
import os
import time
import pandas as pd
from datetime import datetime
from typing import Optional, Dict
import warnings
import requests
import base64
import json
from groq import Groq
from dotenv import load_dotenv, find_dotenv
import nltk
from nltk.corpus import stopwords
from jiwer import wer, cer
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
try:
    _ENV_PATH = find_dotenv(usecwd=True)
    if _ENV_PATH:
        load_dotenv(_ENV_PATH, override=False)
    else:
        load_dotenv(override=False)
except Exception:
    pass

# Ensure NLTK stopwords are available
try:
    _ = stopwords.words('english')
except Exception:
    try:
        nltk.download('stopwords')
    except Exception:
        pass

# Import streamlit-audiorecorder (module name is 'audiorecorder')
try:
    from audiorecorder import audiorecorder as audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except Exception:
    AUDIO_RECORDER_AVAILABLE = False
    st.error("streamlit-audiorecorder not available. Please install it: pip install streamlit-audiorecorder")

# Page config
st.set_page_config(
    page_title="Medical Whisper Live STT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .transcription-box {
        background-color: #ffffff;
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        padding: 1rem;
        min-height: 200px;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .recording-status {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .recordings-table {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .table-info {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Text processing helpers
# ----------------------

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = " ".join(text.split())
    return text.lower()


def remove_stopwords(text: str, language: str = 'english') -> str:
    try:
        stop_set = set(stopwords.words(language))
    except Exception:
        stop_set = set()
    tokens = normalize_text(text).split()
    filtered_tokens = [tok for tok in tokens if tok not in stop_set]
    return " ".join(filtered_tokens)


def preprocess_for_metrics(text: str) -> str:
    return remove_stopwords(text or "")


def compute_wer_cer(ground_truth: str, hypothesis: str) -> Dict[str, float]:
    gt = preprocess_for_metrics(ground_truth)
    hyp = preprocess_for_metrics(hypothesis)
    if len(gt.strip()) == 0 and len(hyp.strip()) == 0:
        return {"wer": 0.0, "cer": 0.0}
    try:
        return {"wer": float(wer(gt, hyp)), "cer": float(cer(gt, hyp))}
    except Exception:
        return {"wer": 1.0, "cer": 1.0}


def compute_cosine_similarity(text_a: str, text_b: str) -> float:
    a = preprocess_for_metrics(text_a)
    b = preprocess_for_metrics(text_b)
    try:
        vec = TfidfVectorizer()
        tfidf = vec.fit_transform([a, b])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def clean_llm_response(text: str) -> str:
    if not text:
        return text
    # Minimal cleanup consistent with metrics script
    import re as _re
    text = _re.sub(r'\*\*.*?\*\*', '', text)
    text = _re.sub(r'\*.*?\*', '', text)
    text = _re.sub(r'^#+\s*', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'^[-*]\s*', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'^[0-9]+\.\s*', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'^Impression:\s*', '', text, flags=_re.IGNORECASE)
    text = _re.sub(r'^Findings:\s*', '', text, flags=_re.IGNORECASE)
    text = _re.sub(r'^Conclusion:\s*', '', text, flags=_re.IGNORECASE)
    text = _re.sub(r'^Here is the cleaned.*?:', '', text, flags=_re.IGNORECASE)
    text = _re.sub(r'^Here is the corrected.*?:', '', text, flags=_re.IGNORECASE)
    text = _re.sub(r'^Here is the.*?:', '', text, flags=_re.IGNORECASE)
    text = _re.sub(r'\n+', ' ', text)
    text = _re.sub(r'\s+', ' ', text).strip()
    return text


def setup_gemini_model():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model_name = os.getenv('DEFAULT_MODEL', 'gemini-2.0-flash-lite')
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            },
        )
        return model
    except Exception:
        return None


def post_process_with_llm(text: str, model) -> str:
    if not text or not model:
        return text or ""
    try:
        prompt = f"""
You are a medical transcription specialist. Please clean and correct the following medical transcription to make it more accurate, professional, and properly formatted for medical reports.

Guidelines:
1. Fix any obvious transcription errors
2. Ensure proper medical terminology
3. Maintain the original meaning and medical findings
4. Use proper medical formatting and punctuation
5. Keep the same structure and content as the original
6. Do not add information that wasn't in the original text
7. Return ONLY the cleaned text without any formatting symbols, bullet points, or section headers
8. Return the text as a single paragraph without line breaks
9. Do not add any introductory text like "Here is the cleaned version" or similar

Original transcription:
{text}

Please provide the cleaned and corrected version as a single paragraph:
"""
        resp = model.generate_content(prompt)
        return clean_llm_response(resp.text.strip())
    except Exception:
        return text

class MedicalWhisperSTT:
    """Medical Whisper Speech-to-Text class for Streamlit"""
    
    def __init__(self):
        self.pipe = None
        self.processor = None
        self.model = None
        self.device = None
        # Multiple model support
        self.pipes = {}
        
    def load_model(self, model_name: str, device: str = "auto"):
        """Load the medical whisper model"""
        try:
            # Determine device
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = device
            
            # Load model with pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                chunk_length_s=30,
                stride_length_s=5
            )
            
            return True, f"Model loaded successfully on {self.device}"
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def load_models(self, model_names, device: str = "auto") -> Dict:
        """Load multiple models for comparison. Returns per-model status."""
        statuses = {}
        # Determine device once
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        for name in model_names:
            if name in self.pipes:
                statuses[name] = (True, "Already loaded")
                continue
            try:
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    chunk_length_s=30,
                    stride_length_s=5
                )
                self.pipes[name] = pipe
                statuses[name] = (True, f"Loaded on {self.device}")
            except Exception as e:
                statuses[name] = (False, f"Error: {e}")
        return statuses
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Transcribe audio data"""
        try:
            if self.pipe is None:
                return {"text": "", "error": "Model not loaded"}
            
            # Transcribe using pipeline
            result = self.pipe(audio_data)
            text = result["text"] if isinstance(result, dict) else result
            
            return {"text": text.strip(), "error": None}
            
        except Exception as e:
            return {"text": "", "error": str(e)}
    
    def transcribe_audio_file(self, audio_bytes: bytes) -> Dict:
        """Transcribe audio from bytes"""
        try:
            if self.pipe is None:
                return {"text": "", "error": "Model not loaded"}
            
            # Save audio bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Load audio
                audio_data, sample_rate = librosa.load(tmp_file_path, sr=16000)
                
                # Transcribe
                result = self.transcribe_audio(audio_data, sample_rate)
                
                return result
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            
        except Exception as e:
            return {"text": "", "error": str(e)}

    def _bytes_to_audio(self, audio_bytes: bytes, target_sr: int = 16000):
        """Convert bytes (wav/other) to mono float32 numpy array and sample rate."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            path = tmp_file.name
        try:
            audio_data, sample_rate = librosa.load(path, sr=target_sr)
            return audio_data, sample_rate
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def transcribe_with_groq_api(self, audio_bytes: bytes, model_name: str) -> Dict:
        """Transcribe audio using Groq API"""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Initialize Groq client
                client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                
                # Open the audio file and create transcription
                with open(tmp_file_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(tmp_file_path, file.read()),
                        model=model_name,
                        response_format="json",
                        temperature=0.0
                    )
                
                return {"text": transcription.text, "error": None}
                    
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            return {"text": "", "error": str(e)}

    def transcribe_with_all(self, audio_bytes: bytes) -> Dict[str, Dict]:
        """Transcribe the same audio with all loaded models. Returns per-model results and timings."""
        results: Dict[str, Dict] = {}
        
        # Process local models
        if self.pipes:
            audio_data, sr = self._bytes_to_audio(audio_bytes, target_sr=16000)
            for name, p in self.pipes.items():
                start_t = time.time()
                try:
                    out = p(audio_data)
                    text = out["text"] if isinstance(out, dict) else out
                    err = None
                except Exception as e:
                    text = ""
                    err = str(e)
                dur = time.time() - start_t
                results[name] = {"text": text.strip(), "error": err, "time_sec": dur}
        
        # Process Groq API models
        groq_models = {
            "groq/whisper-large-v3": "whisper-large-v3",
            "groq/whisper-large-v3-turbo": "whisper-large-v3-turbo"
        }
        
        for model_name, api_model in groq_models.items():
            start_t = time.time()
            try:
                result = self.transcribe_with_groq_api(audio_bytes, api_model)
                text = result.get("text", "")
                err = result.get("error")
            except Exception as e:
                text = ""
                err = str(e)
            dur = time.time() - start_t
            results[model_name] = {"text": text.strip(), "error": err, "time_sec": dur}
        
        return results

def add_recording_to_table(transcription_text, model_results=None, audio_duration=None):
    """Add a recording to the recordings table"""
    # Generate recording ID
    recording_id = f"rec_{len(st.session_state.recordings_table) + 1:03d}"
    
    # Create base entry
    recording_entry = {
        'file_name': recording_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'audio_duration': audio_duration or 'N/A'
    }
    
    # Add model-specific columns
    if model_results:
        for model_name, result in model_results.items():
            # Clean model name for column headers
            clean_model_name = model_name.replace('/', '_').replace('-', '_')
            text_val = result.get('text', '')
            post_val = result.get('post_text', '')
            recording_entry[f'{clean_model_name}_transcript'] = text_val
            recording_entry[f'{clean_model_name}_transcript_processed'] = post_val
            # Metrics if ground truth exists
            gt = st.session_state.get('ground_truth', '')
            if gt:
                m = compute_wer_cer(gt, text_val) if text_val else {"wer": None, "cer": None}
                m2 = compute_wer_cer(gt, post_val) if post_val else {"wer": None, "cer": None}
                cos = compute_cosine_similarity(gt, text_val) if text_val else None
                cos2 = compute_cosine_similarity(gt, post_val) if post_val else None
                recording_entry[f'{clean_model_name}_WER'] = m["wer"]
                recording_entry[f'{clean_model_name}_CER'] = m["cer"]
                recording_entry[f'{clean_model_name}_WER_processed'] = m2["wer"]
                recording_entry[f'{clean_model_name}_CER_processed'] = m2["cer"]
                recording_entry[f'{clean_model_name}_Cosine'] = cos
                recording_entry[f'{clean_model_name}_Cosine_processed'] = cos2
            recording_entry[f'{clean_model_name}_time'] = f"{result.get('time_sec', 0):.2f}s"
    else:
        # If no model results, add a default transcription
        recording_entry['default_transcript'] = transcription_text
        recording_entry['default_time'] = 'N/A'
    
    st.session_state.recordings_table.append(recording_entry)

def get_recordings_dataframe():
    """Convert recordings table to pandas DataFrame"""
    if not st.session_state.recordings_table:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.recordings_table)

# Initialize session state
if 'stt_engine' not in st.session_state:
    st.session_state.stt_engine = MedicalWhisperSTT()
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""
if 'recordings' not in st.session_state:
    st.session_state.recordings = []
if 'recordings_table' not in st.session_state:
    st.session_state.recordings_table = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'metrics_computed' not in st.session_state:
    st.session_state.metrics_computed = False
if 'comparison_rows' not in st.session_state:
    st.session_state.comparison_rows = []

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Whisper Live Speech-to-Text</h1>', unsafe_allow_html=True)
    
    # Check if audio recorder is available
    if not AUDIO_RECORDER_AVAILABLE:
        st.error("❌ streamlit-audiorecorder not available. Please install it:")
        st.code("pip install streamlit-audiorecorder")
        return
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Multi-model selection (for comparison)
        model_options = {
            "Na0s/Medical-Whisper-Large-v3": "Na0s Medical-Whisper-Large-v3",
            "Crystalcareai/Whisper-Medicalv1": "Crystalcareai Whisper-Medicalv1",
            "openai/whisper-large-v3": "OpenAI Whisper Large v3 (Local)",
            "groq/whisper-large-v3": "Groq Whisper Large v3 (API)",
            "groq/whisper-large-v3-turbo": "Groq Whisper Large v3 Turbo (API)"
        }
        
        selected_models = st.multiselect(
            "Choose models for comparison:",
            options=list(model_options.keys()),
            default=["Na0s/Medical-Whisper-Large-v3", "openai/whisper-large-v3", "groq/whisper-large-v3", "groq/whisper-large-v3-turbo"],
            format_func=lambda x: model_options[x],
            help="Select one or more models to compare on the same recording"
        )
        
        # Device selection
        device_options = ["auto", "cpu", "cuda", "mps"]
        selected_device = st.selectbox(
            "Choose Device:",
            options=device_options,
            help="Select the device for model inference"
        )
        
        # Load models button
        if st.button("🔄 Load Selected Models", type="primary"):
            if not selected_models:
                st.warning("Please choose at least one model.")
            else:
                with st.spinner("Loading models..."):
                    # Separate local and API models
                    local_models = [m for m in selected_models if not m.startswith('groq/')]
                    groq_models = [m for m in selected_models if m.startswith('groq/')]
                    
                    statuses = {}
                    
                    # Load local models
                    if local_models:
                        local_statuses = st.session_state.stt_engine.load_models(local_models, selected_device)
                        statuses.update(local_statuses)
                    
                    # Check Groq API models
                    if groq_models:
                        # Check if Groq API key is available
                        if not os.getenv('GROQ_API_KEY'):
                            for groq_model in groq_models:
                                statuses[groq_model] = (False, "Groq API key not found in environment variables")
                        else:
                            for groq_model in groq_models:
                                statuses[groq_model] = (True, "Groq API model ready (requires internet connection)")
                    
                    ok = any(s[0] for s in statuses.values())
                    st.session_state.model_loaded = ok
                    st.session_state.selected_models = selected_models  # Store selected models
                    
                    for name, (success, msg) in statuses.items():
                        if success:
                            st.success(f"{model_options.get(name, name)}: {msg}")
                        else:
                            st.error(f"{model_options.get(name, name)}: {msg}")
        
        # Model status
        if st.session_state.model_loaded:
            st.markdown('<p class="status-success">✅ Models Ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ Model Not Loaded</p>', unsafe_allow_html=True)
        
        # API Key Status
        st.header("🔑 API Configuration")
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            st.success("✅ Groq API key configured")
        else:
            st.warning("⚠️ Groq API key not found. Set GROQ_API_KEY environment variable to use API models.")

        # (Ground truth and Gemini controls moved to main area below)
        
        # Instructions
        st.header("📋 Instructions")
        st.markdown("""
        1. **Load Models**: Click "Load Selected Models" button
        2. **Record Audio**: Use the audio recorder below
        3. **Get Transcription**: Audio will be transcribed automatically
        4. **Edit Text**: Modify the transcription as needed
        5. **Download**: Save the final transcription
        
        **Available Models:**
        - **Local Models**: Run on your device (faster, no internet required)
        - **Groq API Models**: Use Groq's fast cloud service (requires internet and API key)
        """)
    
    # Main content area
    st.header("🎙️ Live Audio Recording")
    
    # Ground truth and Gemini controls in main area
    st.markdown("---")
    st.subheader("🧠 Post-processing & Metrics")
    if 'ground_truth' not in st.session_state:
        st.session_state.ground_truth = ""
    if 'use_gemini_post' not in st.session_state:
        st.session_state.use_gemini_post = False
    if 'gemini_model_ready' not in st.session_state:
        st.session_state.gemini_model_ready = bool(os.getenv('GOOGLE_API_KEY'))

    st.session_state.ground_truth = st.text_area(
        "Ground Truth (optional)",
        value=st.session_state.ground_truth,
        help="Provide ground truth text to compute WER/CER (stopwords removed) and cosine similarity."
    )

    st.session_state.use_gemini_post = st.checkbox(
        "Use Gemini LLM post-processing", value=st.session_state.use_gemini_post,
        help="Requires GOOGLE_API_KEY. Cleans model outputs before metrics."
    )

    if st.session_state.use_gemini_post:
        if not os.getenv('GOOGLE_API_KEY'):
            st.warning("GOOGLE_API_KEY not set; LLM post-processing will be skipped.")
        else:
            st.info("Gemini post-processing enabled")

    # Recording status
    if st.session_state.model_loaded:
        st.markdown("""
        <div class="recording-status">
            <h4>🎤 Ready to Record</h4>
            <p>Click the microphone button below to start recording your voice. 
            The audio will be automatically transcribed when you stop recording.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please load a model first before recording.")
    
    # Audio recorder
    if st.session_state.model_loaded:
        # The audiorecorder component takes two prompt strings (start/stop)
        audio_obj = audio_recorder("🎤 Click to record", "⏹️ Recording... Click to stop")
        
        # Process recorded audio
        if audio_obj is not None:
            st.success("🎵 Audio recorded! Processing...")
            
            # Convert the returned object to WAV bytes
            wav_bytes: bytes = b""
            try:
                if hasattr(audio_obj, "tobytes"):
                    # numpy-like buffer
                    wav_bytes = audio_obj.tobytes()
                elif hasattr(audio_obj, "export"):
                    # pydub.AudioSegment
                    buf = io.BytesIO()
                    audio_obj.export(buf, format="wav")
                    wav_bytes = buf.getvalue()
                elif isinstance(audio_obj, (bytes, bytearray)):
                    wav_bytes = bytes(audio_obj)
                else:
                    # As a last resort, try writing via soundfile if ndarray-like
                    import numpy as _np
                    import soundfile as _sf
                    arr = _np.array(audio_obj)
                    buf = io.BytesIO()
                    _sf.write(buf, arr, 16000, format='WAV')
                    wav_bytes = buf.getvalue()
            except Exception as _e:
                st.error(f"Unable to convert recording to WAV: {_e}")
                wav_bytes = b""
            
            if wav_bytes:
                with st.spinner("Transcribing with all selected models..."):
                    # Get selected models from session state
                    selected_models = st.session_state.get('selected_models', [])
                    if not selected_models:
                        st.error("No models selected. Please load models first.")
                    else:
                        results = {}

                        # Prepare optional Gemini model
                        gemini_model = setup_gemini_model() if st.session_state.use_gemini_post else None

                        # Process local models
                        local_models = [m for m in selected_models if not m.startswith('groq/')]
                        if local_models and st.session_state.stt_engine.pipes:
                            audio_data, sr = st.session_state.stt_engine._bytes_to_audio(wav_bytes, target_sr=16000)
                            for name, p in st.session_state.stt_engine.pipes.items():
                                if name in local_models:
                                    start_t = time.time()
                                    try:
                                        out = p(audio_data)
                                        text = out["text"] if isinstance(out, dict) else out
                                        err = None
                                    except Exception as e:
                                        text = ""
                                        err = str(e)
                                    dur = time.time() - start_t
                                    post_text = post_process_with_llm(text.strip(), gemini_model) if text else ""
                                    results[name] = {
                                        "text": text.strip(),
                                        "post_text": post_text,
                                        "error": err,
                                        "time_sec": dur,
                                    }

                        # Process Groq API models
                        groq_models = [m for m in selected_models if m.startswith('groq/')]
                        for groq_model in groq_models:
                            start_t = time.time()
                            try:
                                # Map Groq model names to actual API model names
                                model_mapping = {
                                    "groq/whisper-large-v3": "whisper-large-v3",
                                    "groq/whisper-large-v3-turbo": "whisper-large-v3-turbo"
                                }
                                api_model_name = model_mapping.get(groq_model, "whisper-large-v3")
                                result = st.session_state.stt_engine.transcribe_with_groq_api(wav_bytes, api_model_name)
                                text = result.get("text", "")
                                err = result.get("error")
                            except Exception as e:
                                text = ""
                                err = str(e)
                            dur = time.time() - start_t
                            post_text = post_process_with_llm(text.strip(), gemini_model) if text else ""
                            results[groq_model] = {
                                "text": text.strip(),
                                "post_text": post_text,
                                "error": err,
                                "time_sec": dur,
                            }

                        if not results:
                            st.error("No models loaded.")
                        else:
                            # Store raw results for later metric computation
                            st.session_state.last_results = results
                            # Show a preview table with outputs and times only
                            preview_rows = []
                            for name, out in results.items():
                                label = model_options.get(name, name)
                                preview_rows.append({
                                    "Model": label,
                                    "Output": out.get("text", ""),
                                    "Output (Processed)": out.get("post_text", ""),
                                    "Time": f"{out.get('time_sec', 0.0):.2f}s",
                                    "Status": "OK" if out.get("text") else "ERR",
                                })
                            st.subheader("🧪 Model Outputs (Preview)")
                            st.table(pd.DataFrame(preview_rows))
                            st.info("Provide Ground Truth and click Compute to calculate metrics.")
    
    # Transcription section
    st.header("📝 Live Transcription")
    
    # Transcription text area
    transcription_text = st.text_area(
        "Transcribed Text:",
        value=st.session_state.transcription_text,
        height=300,
        help="Your transcribed text appears here. You can edit it as needed.",
        key="transcription_area"
    )
    
    # Update session state when text changes
    if transcription_text != st.session_state.transcription_text:
        st.session_state.transcription_text = transcription_text
    
    # Action buttons: Compute, Add to recording table, Reset
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        compute_disabled = not (st.session_state.ground_truth.strip() and st.session_state.last_results)
        if st.button("🧮 Compute", disabled=compute_disabled, type="primary"):
            rows = []
            gt = st.session_state.ground_truth
            results = st.session_state.last_results or {}
            for name, out in results.items():
                label = name
                text = out.get("text", "")
                post_text = out.get("post_text", "")
                tsec = out.get("time_sec", 0.0)
                wer_val = cer_val = cos_val = None
                wer_post = cer_post = cos_post = None
                if gt and text:
                    m = compute_wer_cer(gt, text)
                    wer_val, cer_val = m["wer"], m["cer"]
                    cos_val = compute_cosine_similarity(gt, text)
                if gt and post_text:
                    m2 = compute_wer_cer(gt, post_text)
                    wer_post, cer_post = m2["wer"], m2["cer"]
                    cos_post = compute_cosine_similarity(gt, post_text)
                rows.append({
                    "Model": label,
                    "Output": text,
                    "Output (Processed)": post_text,
                    "Time": f"{tsec:.2f}s",
                    "WER": wer_val,
                    "CER": cer_val,
                    "Cosine": cos_val,
                    "WER (Processed)": wer_post,
                    "CER (Processed)": cer_post,
                    "Cosine (Processed)": cos_post,
                })
            st.session_state.comparison_rows = rows
            st.session_state.metrics_computed = True

    with col2:
        add_disabled = not (st.session_state.metrics_computed and st.session_state.comparison_rows and st.session_state.last_results)
        if st.button("➕ Add to recording table", disabled=add_disabled):
            # Use first successful text as editable default if needed
            editable_text = None
            for name, out in (st.session_state.last_results or {}).items():
                candidate = out.get("text")
                if candidate:
                    editable_text = candidate
                    break
            if editable_text:
                add_recording_to_table(editable_text, st.session_state.last_results)
                st.success("✅ Added to recordings table!")
            else:
                st.warning("⚠️ No valid transcript to add.")

    with col3:
        if st.button("🧹 Reset"):
            st.session_state.ground_truth = ""
            st.session_state.transcription_text = ""
            st.session_state.last_results = None
            st.session_state.metrics_computed = False
            st.session_state.comparison_rows = []
            st.rerun()

    with col4:
        if st.button("📋 Copy Text"):
            st.code(st.session_state.transcription_text)

    with col5:
        if st.button("💾 Download"):
            if st.session_state.transcription_text:
                st.download_button(
                    label="Download Transcription",
                    data=st.session_state.transcription_text,
                    file_name="live_transcription.txt",
                    mime="text/plain"
                )
    
    # Show computed metrics table if available
    if st.session_state.metrics_computed and st.session_state.comparison_rows:
        st.subheader("📐 Metrics (Computed)")
        st.table(pd.DataFrame(st.session_state.comparison_rows))

    # Recordings Table Section
    st.markdown("---")
    st.markdown('<div class="recordings-table">', unsafe_allow_html=True)
    st.header("📊 Recordings Table")
    
    if st.session_state.recordings_table:
        # Display the table
        df = get_recordings_dataframe()
        
        # Format the dataframe for better display
        display_df = df.copy()
        if not display_df.empty:
            # Truncate transcript columns for display
            transcript_columns = [col for col in display_df.columns if col.endswith('_transcript') or col.endswith('_transcript_processed')]
            for col in transcript_columns:
                display_df[col] = display_df[col].apply(
                    lambda x: x[:50] + '...' if len(str(x)) > 50 else str(x)
                )
        
        st.dataframe(display_df, use_container_width=True)
        
        # Show full transcriptions in expandable sections
        st.subheader("📋 Full Transcriptions")
        for i, recording in enumerate(st.session_state.recordings_table):
            with st.expander(f"Recording {recording['file_name']} - {recording['timestamp']}"):
                # Show all model transcriptions
                transcript_columns = [col for col in recording.keys() if col.endswith('_transcript')]
                for col in transcript_columns:
                    model_name = col.replace('_transcript', '')
                    st.write(f"**{model_name}**: {recording[col]}")
                    time_col = f"{model_name}_time"
                    if time_col in recording:
                        st.caption(f"Processing time: {recording[time_col]}")
        
        # Table actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"recordings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Clear table
            if st.button("🗑️ Clear Table"):
                st.session_state.recordings_table = []
                st.rerun()
        
        with col3:
            # Show table info
            st.markdown(f'<div class="table-info">📈 Total recordings: {len(st.session_state.recordings_table)}</div>', unsafe_allow_html=True)
    else:
        st.info("📝 No recordings yet. Start recording to see them here!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Alternative file upload section
    st.markdown("---")
    st.header("🎵 Alternative: Upload Audio File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        help="Upload an audio file for transcription (alternative to live recording)"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size / 1024:.1f} KB")
        
        # Transcribe button
        if st.button("🎙️ Transcribe Uploaded File", type="primary", disabled=not st.session_state.model_loaded):
            if st.session_state.model_loaded:
                with st.spinner("Transcribing audio..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Load audio
                        audio_data, sample_rate = librosa.load(tmp_file_path, sr=16000)
                        
                        # Transcribe with all models
                        import soundfile as _sf
                        buf = io.BytesIO()
                        _sf.write(buf, audio_data, 16000, format='WAV')
                        wav_bytes = buf.getvalue()
                        results = st.session_state.stt_engine.transcribe_with_all(wav_bytes)
                        if not results:
                            st.error("No models loaded.")
                        else:
                            rows = []
                            editable_text = None
                            for name, out in results.items():
                                label = model_options.get(name, name)
                                text = out.get("text", "")
                                err = out.get("error")
                                tsec = out.get("time_sec", 0.0)
                                rows.append((label, text if text else (err or ""), f"{tsec:.2f}s", "OK" if text else "ERR"))
                                if editable_text is None and text:
                                    editable_text = text
                            st.subheader("🧪 Model Comparison")
                            st.table({
                                "Model": [r[0] for r in rows],
                                "Output": [r[1] for r in rows],
                                "Time": [r[2] for r in rows],
                                "Status": [r[3] for r in rows],
                            })
                            if editable_text:
                                st.session_state.transcription_text = editable_text
                                # Add to recordings table
                                add_recording_to_table(editable_text, results)
                                st.success("✅ Comparison completed! Edit the text on the right.")
                            else:
                                st.warning("⚠️ None of the models produced text.")
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
            else:
                st.error("Please load a model first!")
    
    # Recording history
    if st.session_state.recordings:
        st.markdown("---")
        st.header("📚 Recording History")
        
        for i, recording in enumerate(reversed(st.session_state.recordings[-5:])):  # Show last 5
            with st.expander(f"Recording {len(st.session_state.recordings) - i} - {time.strftime('%H:%M:%S', time.localtime(recording['timestamp']))}"):
                st.text(recording['text'])
    
    # Footer with model info
    if st.session_state.model_loaded:
        st.markdown("---")
        st.markdown(f"""
        <div class="model-info">
            <h4>🤖 Model Information</h4>
            <p><strong>Models:</strong> {', '.join(selected_models) if 'selected_models' in locals() else ''}</p>
            <p><strong>Device:</strong> {st.session_state.stt_engine.device}</p>
            <p><strong>Status:</strong> <span class="status-success">Ready for comparison</span></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
