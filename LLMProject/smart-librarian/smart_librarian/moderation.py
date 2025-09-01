# smart_librarian/moderation.py
BAD_WORDS = {
    "shit", "idiot", "stupid", "bastard" #extendable list
}

def is_offensive(text: str) -> bool:
    """A simple filter that returns True if any forbidden word appears."""
    if not text:
        return False
    t = text.lower()
    return any(bad in t for bad in BAD_WORDS)
