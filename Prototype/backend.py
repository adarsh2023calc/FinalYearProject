# transcribe_backends.py
import os
import tempfile
from typing import Dict, Any, Optional

# Backend for AI
def transcribe_with_whisper_local(wav_path: str, language: Optional[str] = None, model_name: str = "small"):
    """
    Transcribe a local WAV file using the whisper package.
    language: ISO code like 'en', 'hi' (None = auto-detect)
    model_name: "tiny", "base", "small", "medium", "large"
    """
    try:
        import whisper
    except Exception as e:
        raise RuntimeError("Please install the 'whisper' package: pip install -U openai-whisper") from e

    model = whisper.load_model(model_name)
    options = {"beam_size": 5}
    if language:
        options["language"] = language
        options["task"] = "transcribe"
    
    
    result = model.transcribe(wav_path, **options)
    return {"text": result.get("text", ""), "raw": result}

# OpenAI speech-to-text backend (if you prefer OpenAI cloud)
def transcribe_with_openai_api(wav_path: str, language: Optional[str] = None, api_key: Optional[str] = None, model: str = "gpt-4o-transcribe"):
    """
    Transcribe a WAV file using OpenAI Speech-to-Text API.
    NOTE: API names and usage may change — check OpenAI docs for the exact call.
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("Please install the 'openai' package: pip install openai") from e

    if api_key:
        openai.api_key = api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key.")

    # Read file bytes
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    # Example usage: openai.Audio.transcribe (names may differ; adapt to latest SDK)
    # This is a generic example; if your SDK uses different call parameters update accordingly.
    resp = openai.Audio.transcribe(model=model, file=audio_bytes, language=language)
    # resp usually contains 'text' or 'transcript' depending on SDK
    text = resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", None)
    return {"text": text or "", "raw": resp}
