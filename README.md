# Search Engine Bot (Hugging Face demo)

This is a small FastAPI demo that accepts an audio upload and uses the Hugging Face Inference API to transcribe audio and translate it to English. Results are stored in `database.json`.

Quick start

1. Install dependencies (prefer a virtualenv):

```powershell
python -m pip install -r requirements.txt
```

2. Set environment variables (example):

```powershell
$Env:HF_API_TOKEN = "<your-hf-token>"
# optional: choose another model
$Env:HF_MODEL = "openai/whisper-small"
```

3. Run the server:

```powershell
python -m uvicorn main:app --reload --port 8000
```

4. Send a file (example with `curl`):

```powershell
curl -X POST "http://127.0.0.1:8000/talk" -F "file=@path\to\audio.wav"
```

Notes
- This example uses the Hugging Face Inference API (`huggingface-hub`). You need an API token (free-tier available on HF).
- The code attempts to store both the original transcript and an English translation in `database.json`.
- For a quick demo, use `openai/whisper-small` as the default model. You can switch models by setting `HF_MODEL`.