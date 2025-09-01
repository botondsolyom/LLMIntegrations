from pathlib import Path
from typing import Optional
from openai import OpenAI

# From TTS to mp3
def synthesize_speech(text: str, out_path: Path, voice: str = "alloy",
                      model: str = "gpt-4o-mini-tts") -> Optional[Path]:
    """
    Generate speech audio (mp3) from text using OpenAI TTS models.
    Returns the path if successful, else None.
    """
    if not text or not text.strip():
        return None

    client = OpenAI()
    # The .create() returns byte; save in mp3
    resp = client.audio.speech.create(
        model=model,       # or: "tts-1" / "tts-1-hd"
        voice=voice,
        input=text.strip()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(resp.content)   # mp3 bytes
    return out_path
