# streamlit_app.py
import os
from pathlib import Path
from typing import List, Dict, Any
import json

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# internal modules
from smart_librarian.rag import search_books
from smart_librarian.tools import get_summary_by_title, OPENAI_TOOLS
from smart_librarian.moderation import is_offensive
from io import BytesIO
from smart_librarian.tts import synthesize_speech
from streamlit_mic_recorder import mic_recorder
import tempfile
from pathlib import Path

from smart_librarian.stt import transcribe_audio




# --- env / client ---
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
client = OpenAI()

SYSTEM_PROMPT = (
    "You are Smart Librarian. Based on the user's interests and the provided candidate books, "
    "recommend exactly ONE book title (an existing title from the candidates). Then, if possible, "
    "call the function get_summary_by_title(title) with that exact title so we can show the full summary.\n"
    "Keep your tone friendly and concise."
)

def build_context_snippet(hits: List[Dict[str, Any]]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}) {h['title']}: {h['document'][:400]}")
    return "\n".join(lines)

def llm_recommend_with_tool(user_query: str, hits: List[Dict[str, Any]]) -> str:
    """LLM recommend + (if is necessary) it calls the local tool and eventually returns a final response."""
    context = build_context_snippet(hits)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User query: {user_query}\n"
                f"Candidate books (from RAG):\n{context}\n"
                f"Pick ONE title from the candidates and then call the tool."
            )
        },
    ]

    first = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=OPENAI_TOOLS,
        tool_choice="auto",
        temperature=0.3,
    )

    msg = first.choices[0].message
    tool_calls = msg.tool_calls or []
    tool_messages = []

    if tool_calls:
        for tc in tool_calls:
            if tc.function.name == "get_summary_by_title":
                args = json.loads(tc.function.arguments or "{}")
                title = args.get("title", "")
                summary = get_summary_by_title(title) or "(No detailed summary available.)"
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "get_summary_by_title",
                    "content": summary,
                })

        final = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages + [msg] + tool_messages,
            temperature=0.2,
        )
        return final.choices[0].message.content or ""

    # If the tool wasn't called we at least return the first recommendation
    return msg.content or ""


# ---------------- UI ----------------
st.set_page_config(page_title="Smart Librarian", page_icon="📚", layout="centered")
st.title("📚 Smart Librarian — RAG + Tool")

with st.sidebar:
    st.subheader("Settings")
    st.caption("Models from .env")
    st.write(f"**CHAT_MODEL**: `{CHAT_MODEL}`")
    st.write(f"**EMBED_MODEL**: `{EMBED_MODEL}`")
    st.divider()
    st.caption("Tip: add `CHROMA_TELEMETRY_ENABLED=false` to .env to silence red telemetry lines in console.")

st.write("Tell me what you like, and I'll recommend a book. Example: _“I want a book about freedom and social control.”_")

query = st.text_input("What would you like to read about?")

#STT
st.write("---")
st.subheader("🎤 Voice Mode (microphone)")

st.caption("Click *Record*, say your query, then click *Stop*. We'll transcribe it and use it as your question.")

# 1) Microphone recording (from browser)
rec = mic_recorder(
    start_prompt="🎙️ Record",
    stop_prompt="🛑 Stop",
    just_once=False,
    key="mic",
)

# 2) If there is a recording → save it to a temporary file and transcribe it
if rec is not None:
    # Returns WAV bytes, mostly
    audio_bytes = None
    sample_rate = None

    if isinstance(rec, dict):
        # mostly: {"bytes": b"...wav...", "sample_rate": 16000, ...}
        audio_bytes = rec.get("bytes", None)
        sample_rate = rec.get("sample_rate", None)
    elif isinstance(rec, (bytes, bytearray)):
        audio_bytes = bytes(rec)

    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
            tmpf.write(audio_bytes)
            tmp_path = Path(tmpf.name)

        st.info("Transcribing your voice input…")
        transcript = transcribe_audio(tmp_path)  # -> string or None

        if transcript:
            st.success(f"🗣️ You said: {transcript}")
            # Fill the question field
            query = transcript
        else:
            st.error("❌ Could not transcribe audio. Try again.")
    else:
        st.warning("Could not read audio bytes from recorder result. Please update 'streamlit-mic-recorder' or try again.")

go = st.button("Recommend")

speak = st.checkbox("🔊 Also speak the recommendation (TTS)")

if go and query.strip():
    if is_offensive(query):
        st.warning("⚠️ Please rephrase politely. Offensive language is not allowed.")
    else:
        with st.spinner("Searching your library…"):
            hits = search_books(query, k=3)

        if not hits:
            st.warning("I couldn't find relevant books. Try rephrasing your interests.")
        else:
            with st.expander("Candidate books (from RAG)", expanded=False):
                for i, h in enumerate(hits, 1):
                    st.markdown(f"**{i}. {h['title']}**  \n_{h['document'][:500]}…_  \n(distance: {h['distance']:.3f})")

            with st.spinner("Asking the librarian…"):
                answer = llm_recommend_with_tool(query, hits)

            st.markdown("### Recommendation")
            st.write(answer.strip())

            # --- TTS block
            if speak and answer.strip():
                tmp_path = Path("out/audio/last_recommendation.mp3")
                res = synthesize_speech(answer, tmp_path)
                if res and res.exists():
                    audio_bytes = tmp_path.read_bytes()
                    st.audio(audio_bytes, format="audio/mp3")

