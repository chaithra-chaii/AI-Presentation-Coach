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



INFORMAL_WORDS = {
    "gonna", "wanna", "gotta", "kinda", "sorta", "stuff",
    "guys", "dude", "awesome", "lots", "tons", "yep", "nope",
}


def analyze_vocabulary(text, audience="general"):
    """Check whether the language suits the audience.

    audience can be "casual", "general" or "formal".
    """
    words = re.findall(r"[a-z']+", text.lower())

    if not words:
        return {
            "total_words": 0,
            "average_word_length": 0,
            "long_word_ratio": 0,
            "variety": 0,
            "informal_words": [],
            "level": "none",
            "notes": [],
        }

    average_length = round(sum(len(w) for w in words) / len(words), 2)
    long_words = [w for w in words if len(w) >= 9]
    long_ratio = round(len(long_words) / len(words), 2)
    variety = round(len(set(words)) / len(words), 2)
    informal = sorted({w for w in words if w in INFORMAL_WORDS})

    if average_length >= 5.5 or long_ratio >= 0.15:
        level = "advanced"
    elif average_length >= 4.3:
        level = "medium"
    else:
        level = "simple"

    notes = []

    if informal and audience in ("general", "formal"):
        notes.append(
            "Replace informal words (" + ", ".join(informal)
            + ") with more professional wording."
        )

    if len(words) >= 30 and variety < 0.5:
        notes.append("Try using a wider range of words.")

    if audience == "formal" and level == "simple":
        notes.append("Use more precise vocabulary for a formal audience.")

    if audience == "general" and level == "advanced":
        notes.append(
            "Some words may be hard for a general audience. "
            "Use simpler words where you can."
        )

    return {
        "total_words": len(words),
        "average_word_length": average_length,
        "long_word_ratio": long_ratio,
        "variety": variety,
        "informal_words": informal,
        "level": level,
        "notes": notes,
    }



GRAMMAR_RULES = [
    (r"\b(he|she|it)\s+(don't|dont)\b",
     "Use 'doesn't' after he, she or it."),
    (r"\b(we|you|they)\s+is\b",
     "Use 'are' with we, you or they."),
    (r"\bi\s+is\b",
     "Use 'am' with I."),
    (r"\b(he|she|it)\s+are\b",
     "Use 'is' with he, she or it."),
    (r"\b(dont|cant|wont|doesnt|didnt|isnt|arent)\b",
     "This word is missing an apostrophe."),
    (r"\bmore\s+(better|faster|easier|bigger|clearer)\b",
     "Do not use 'more' with a comparative word."),
    (r"\balot\b",
     "'alot' should be 'a lot'."),
]

CONSONANT_SOUND_WORDS = (
    "univers", "user", "use", "usual", "unit", "uniq", "one", "eu",
)
VOWEL_SOUND_WORDS = ("hour", "honest", "honor", "heir")


def check_grammar(text):
    """Find common grammar mistakes using simple rules.

    This is approximate and does not catch every mistake.
    """
    issues = []
    lower = text.lower()

    # Repeated words such as "the the"
    for match in re.finditer(r"\b([a-z']+)\s+\1\b", lower):
        word = match.group(1)
        if word in ("that", "had"):
            continue
        issues.append({
            "text": match.group(0),
            "issue": "Repeated word '" + word + "'.",
            "suggestion": "Remove the extra '" + word + "'.",
        })

    # Lowercase "i" used as a word
    for match in re.finditer(r"(?<![\w'])i(?![\w'])", text):
        issues.append({
            "text": "i",
            "issue": "'I' should be a capital letter.",
            "suggestion": "Write 'I' with a capital letter.",
        })

    # a / an
    for match in re.finditer(r"\b(a|an)\s+([a-z]+)", lower):
        article, word = match.group(1), match.group(2)
        needs_an = word[0] in "aeiou"
        if word.startswith(CONSONANT_SOUND_WORDS):
            needs_an = False
        if word.startswith(VOWEL_SOUND_WORDS):
            needs_an = True
        if article == "a" and needs_an:
            issues.append({
                "text": match.group(0),
                "issue": "Use 'an' before '" + word + "'.",
                "suggestion": "Write 'an " + word + "'.",
            })
        elif article == "an" and not needs_an:
            issues.append({
                "text": match.group(0),
                "issue": "Use 'a' before '" + word + "'.",
                "suggestion": "Write 'a " + word + "'.",
            })

    # Other common mistakes
    for pattern, suggestion in GRAMMAR_RULES:
        for match in re.finditer(pattern, lower):
            issues.append({
                "text": match.group(0),
                "issue": "Possible grammar mistake.",
                "suggestion": suggestion,
            })

    return {"issue_count": len(issues), "issues": issues}