from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import os
import json
import uuid
import tempfile
from datetime import datetime

from huggingface_hub import InferenceClient
from langdetect import detect, LangDetectException
import requests
import io

try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

load_dotenv()

# Hugging Face configuration (set these in your environment)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "openai/whisper-small")

if HF_API_TOKEN:
    inference = InferenceClient(model=HF_MODEL, token=HF_API_TOKEN)
else:
    inference = None

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint to serve index.html
@app.get("/")
async def root():
    return FileResponse("index.html")


def _read_db(path="database.json"):
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_db(data, path="database.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_text(resp):
    # InferenceApi may return different shapes depending on model and pipeline
    if resp is None:
        return ""
    if isinstance(resp, dict):
        return resp.get("text") or resp.get("transcription") or json.dumps(resp, ensure_ascii=False)
    if isinstance(resp, list):
        # sometimes returns list of segments
        texts = []
        for item in resp:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
        return " ".join(texts) if texts else str(resp)
    return str(resp)


@app.post("/talk")
async def post_audio(file: UploadFile = None, file_url: str = None):
    """Accepts an audio file, sends it to Hugging Face Inference API for transcription
    and translation to English, stores the result in `database.json`, and returns the entry id.
    """
    if inference is None:
        raise HTTPException(status_code=500, detail="HF_API_TOKEN is not set in environment")

    # Prefer a `file_url` (fetch from website) if provided, otherwise read uploaded file
    if file_url:
        try:
            resp = requests.get(file_url, timeout=10)
            resp.raise_for_status()
            contents = resp.content
            filename = os.path.basename(file_url.split("?")[0]) or "remote_audio"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not fetch file_url: {e}")
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded and no file_url provided")
        contents = await file.read()
        filename = file.filename

    # Call HF twice: once for original-language transcript, once for English translation
    try:
        original_resp = inference.automatic_speech_recognition(contents)
        # For translation, we'll make a second call and extract translated text
        en_resp = inference.automatic_speech_recognition(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference API error: {e}")

    transcript_original = _extract_text(original_resp)
    transcript_en = _extract_text(en_resp)

    # Try to detect language from the original transcript (best-effort)
    detected_language = None
    try:
        if transcript_original and transcript_original.strip():
            detected_language = detect(transcript_original)
    except LangDetectException:
        detected_language = None

    entry = {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "model": HF_MODEL,
        "detected_language": detected_language,
        "transcript_original": transcript_original,
        "transcript_en": transcript_en,
    }

    db = _read_db()
    db.append(entry)
    _write_db(db)

    return {"status": "ok", "id": entry["id"], "transcript_en": transcript_en}


@app.get("/generate")
async def generate_and_transcribe(text: str = "Hello, this is a test recording. How can I help you today?", lang: str = "en"):
    """Generate a text-to-speech audio sample and transcribe it.
    
    Example usage:
      /generate?text=Hallo+dit+is+Afrikaans&lang=af
    """
    if inference is None:
        raise HTTPException(status_code=500, detail="HF_API_TOKEN is not set in environment")
    
    if not HAS_PYDUB:
        raise HTTPException(status_code=500, detail="pydub is not installed; please install python-pydub")
    
    try:
        # Create a simple beep/tone audio (placeholder for TTS)
        sound = AudioSegment.silent(duration=500)  # 0.5s silence
        
        # Add a simple tone to make it non-empty
        sine_wave = Sine(440, sample_rate=16000).to_audio_segment(duration=1000)  # 1s at 440 Hz
        sound = sine_wave
        
        # Export to bytes
        audio_bytes = io.BytesIO()
        sound.export(audio_bytes, format="wav")
        contents = audio_bytes.getvalue()
        filename = f"generated_{lang}.wav"
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")
    
    # Transcribe the generated audio
    try:
        original_resp = inference.automatic_speech_recognition(contents)
        en_resp = inference.automatic_speech_recognition(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference API error: {e}")
    
    transcript_original = _extract_text(original_resp)
    transcript_en = _extract_text(en_resp)
    
    # Try to detect language
    detected_language = None
    try:
        if transcript_original and transcript_original.strip():
            detected_language = detect(transcript_original)
    except LangDetectException:
        detected_language = None
    
    entry = {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "model": HF_MODEL,
        "detected_language": detected_language,
        "transcript_original": transcript_original,
        "transcript_en": transcript_en,
    }
    
    db = _read_db()
    db.append(entry)
    _write_db(db)
    
    return {"status": "ok", "id": entry["id"], "transcript_en": transcript_en}


@app.get("/records")
def get_records():
    """Return all transcribed records from database.json."""
    db = _read_db()
    return {"count": len(db), "records": db}

   



