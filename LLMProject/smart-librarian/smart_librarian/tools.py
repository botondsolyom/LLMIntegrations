# smart_librarian/tools.py
import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_JSON = PROJECT_ROOT / "data" / "full_summaries.json"

# 1) Python function – the "tool" business logic
def get_summary_by_title(title: str) -> Optional[str]:
    """
    Return the full summary for an exact book title (case-insensitive).
    Data source: data/full_summaries.json
    """
    if not FULL_JSON.exists():
        return None
    data = json.loads(FULL_JSON.read_text(encoding="utf-8"))
    # case-insensitive match
    for k, v in data.items():
        if k.strip().lower() == title.strip().lower():
            return v
    return None


# 2) OpenAI "tools" (function calling) schema – for chat.completions
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Return the detailed summary for an exact book title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact book title to look up."
                    }
                },
                "required": ["title"],
                "additionalProperties": False
            },
        },
    }
]
