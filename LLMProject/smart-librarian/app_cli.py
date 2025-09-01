# app_cli.py
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from smart_librarian.rag import search_books
from smart_librarian.tools import get_summary_by_title, OPENAI_TOOLS
from smart_librarian.moderation import is_offensive
from pathlib import Path
from smart_librarian.tts import synthesize_speech



# --- env / client ---
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")
client = OpenAI()

SYSTEM_PROMPT = (
    "You are Smart Librarian. Based on the user's interests and the provided candidate books, "
    "recommend exactly ONE book title (an existing title from the candidates). Then, if possible, "
    "call the function get_summary_by_title(title) with that exact title so we can show the full summary.\n"
    "Keep your tone friendly and concise. Return ONLY one title."
)

def build_context_snippet(hits: List[Dict[str, Any]]) -> str:
    """Little text context from RAG for the model."""
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}) {h['title']}: {h['document'][:300]}")  # first ~300 characters are enough
    return "\n".join(lines)

def chat_once(user_query: str) -> None:
    # 1) RAG – top matches
    hits = search_books(user_query, k=3)
    if not hits:
        print("Sorry, I couldn't find relevant books. Try rephrasing your interests.")
        return

    context = build_context_snippet(hits)

    # 2) first round – the model recommends and (as expected) wants to call a tool
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

    # 3) If there is a tool call → we process it locally and then return the result to the model
    tool_calls = msg.tool_calls or []
    tool_messages = []
    if tool_calls:
        for tc in tool_calls:
            if tc.function.name == "get_summary_by_title":
                import json
                args = json.loads(tc.function.arguments or "{}")
                title = args.get("title", "")
                summary = get_summary_by_title(title) or "(No detailed summary available.)"
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "get_summary_by_title",
                    "content": summary,
                })

        # 4) Final response after the tool output
        final = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages + [msg] + tool_messages,
            temperature=0.2,
        )
        answer = final.choices[0].message.content or ""
    else:
        # If for some reason it wasn't called, at least show the first
        answer = msg.content or ""

    # 5) Displaying
    print("\n=== Smart Librarian ===")
    print(answer.strip())
    print("=======================\n")


if __name__ == "__main__":
    print("Smart Librarian (CLI). Type 'exit' to quit.")
    while True:
        q = input("\nWhat would you like to read about? > ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        # filter
        if is_offensive(q):
            print("\n⚠️ Please rephrase politely. I cannot process offensive language.\n")
            continue

        # optional TTS – mp3
        audio_path = synthesize_speech(answer, Path("out/audio/recommendation.mp3"))
        if audio_path:
            print(f"(Saved audio to: {audio_path})")

        if not q or q.lower() in {"exit", "quit"}:
            break
        chat_once(q)
