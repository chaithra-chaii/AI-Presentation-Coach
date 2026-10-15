import re

EXAMPLE_MARKERS = [
    "for example", "for instance", "such as", "imagine",
    "consider", "in fact", "research shows", "according to",
    "case study", "like when",
]


def split_sentences(text):
    """Split a speech into sentences."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def find_unclear_sentences(text, max_words=25):
    """Find sentences that are too long to follow easily."""
    unclear = []
    for sentence in split_sentences(text):
        word_count = len(re.findall(r"\b[\w']+\b", sentence))
        if word_count > max_words:
            unclear.append({
                "sentence": sentence,
                "word_count": word_count,
                "reason": "This sentence is long. Try splitting it into two.",
            })
    return unclear


def check_supporting_examples(text):
    """Check whether the speech uses examples, facts or numbers."""
    lower = text.lower()
    markers_found = [m for m in EXAMPLE_MARKERS if m in lower]
    has_numbers = bool(re.search(r"\d", text))
    return {
        "has_examples": bool(markers_found) or has_numbers,
        "markers_found": markers_found,
        "has_numbers": has_numbers,
    }