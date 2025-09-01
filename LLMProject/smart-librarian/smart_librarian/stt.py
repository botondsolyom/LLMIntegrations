from pathlib import Path
from typing import Optional
from openai import OpenAI
import mimetypes

def transcribe_audio(file_path: Path, model: str = "whisper-1") -> Optional[str]:
    """
    Transcribe a spoken audio file to text using OpenAI STT.
    Works reliably with 'whisper-1' for file uploads (wav/mp3/m4a/ogg/webm).
    """
    client = OpenAI()

    if not file_path.exists() or file_path.stat().st_size == 0:
        return None

    # MIME type
    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime:
        mime = "audio/wav"

    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=(file_path.name, f, mime),
            response_format="text"
        )

    text = getattr(resp, "text", None)
    if isinstance(resp, str) and not text:
        text = resp
    return text.strip() if text else None
