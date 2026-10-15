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



TRANSITION_PHRASES = {
    "sequence": [
        "first", "firstly", "second", "secondly", "third", "thirdly",
        "next", "then", "after that", "afterwards", "finally", "lastly",
    ],
    "addition": [
        "also", "in addition", "moreover", "furthermore", "besides",
    ],
    "contrast": [
        "however", "but", "on the other hand", "although", "whereas",
    ],
    "cause": [
        "because", "therefore", "as a result", "thus", "for this reason",
    ],
    "example": [
        "for example", "for instance",
    ],
    "summary": [
        "in conclusion", "to sum up", "to wrap up", "in summary",
        "to summarize", "to summarise",
    ],
}


def check_transitions(text):
    """Check whether the speaker connects ideas with transition words."""
    lower = text.lower()
    sentence_count = len(split_sentences(text))

    found = {}
    total = 0
    for group, phrases in TRANSITION_PHRASES.items():
        for phrase in phrases:
            pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
            count = len(re.findall(pattern, lower))
            if count:
                found[phrase] = count
                total += count

    notes = []
    if sentence_count < 3:
        level = "too short to judge"
    elif total == 0:
        level = "none"
        notes.append(
            "Connect your ideas with transition phrases such as "
            "'First', 'Next', 'However' and 'Finally'."
        )
    elif total / sentence_count < 0.25:
        level = "few"
        notes.append(
            "Add a few more transition phrases so your points flow "
            "smoothly from one to the next."
        )
    else:
        level = "good"

    return {
        "transitions_found": found,
        "total": total,
        "sentence_count": sentence_count,
        "level": level,
        "notes": notes,
    }



ORDER_INTRO_PHRASES = [
    "hello", "hi everyone", "good morning", "good afternoon",
    "good evening", "welcome", "my name is", "my topic",
    "i will talk", "i want to talk", "i will explain",
    "i will present", "i am going to talk", "let me start",
]

ORDER_CONCLUSION_PHRASES = [
    "in conclusion", "to conclude", "to sum up", "to wrap up",
    "in summary", "to summarize", "to summarise", "the key message",
    "key takeaway", "main takeaway",
]

SEQUENCE_RANK = {
    "first": 1, "firstly": 1,
    "second": 2, "secondly": 2, "next": 2,
    "third": 3, "thirdly": 3,
    "finally": 4, "lastly": 4,
}


def _phrase_in(sentence, phrases):
    lower = sentence.lower()
    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        if re.search(pattern, lower):
            return True
    return False


def check_logical_order(text):
    """Check whether the ideas are presented in a sensible order."""
    sentences = split_sentences(text)
    count = len(sentences)

    if count < 3:
        return {
            "order_ok": True,
            "level": "too short to judge",
            "issues": [],
        }

    half = count / 2
    issues = []

    intro_positions = [
        i for i, s in enumerate(sentences)
        if _phrase_in(s, ORDER_INTRO_PHRASES)
    ]
    if intro_positions and intro_positions[0] >= half:
        issues.append(
            "Your introduction appears late. Start by greeting the "
            "audience and stating your topic."
        )

    conclusion_positions = [
        i for i, s in enumerate(sentences)
        if _phrase_in(s, ORDER_CONCLUSION_PHRASES)
    ]
    if conclusion_positions and conclusion_positions[0] < half:
        issues.append(
            "Your conclusion appears too early. Move the summary to "
            "the end of your speech."
        )

    ranks = []
    for sentence in sentences:
        for word in re.findall(r"[a-z]+", sentence.lower()):
            if word in SEQUENCE_RANK:
                ranks.append(SEQUENCE_RANK[word])
    if ranks != sorted(ranks):
        issues.append(
            "Your 'first / next / finally' points are out of order. "
            "Check the order of your main points."
        )

    return {
        "order_ok": not issues,
        "level": "good" if not issues else "needs work",
        "issues": issues,
    }



IDEA_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "with",
    "this", "that", "it", "i", "we", "you", "they", "he", "she",
    "my", "our", "your", "their", "have", "has", "had", "do", "does",
    "did", "will", "would", "can", "could", "should", "from", "as",
    "by", "about", "so", "also", "very", "just",
}


def _idea_words(sentence):
    words = re.findall(r"[a-z']+", sentence.lower())
    return {w for w in words if w not in IDEA_STOP_WORDS and len(w) >= 2}


def check_repeated_ideas(text, threshold=0.6):
    """Find sentences that say almost the same thing."""
    sentences = split_sentences(text)
    word_sets = [_idea_words(s) for s in sentences]

    repeated = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            first, second = word_sets[i], word_sets[j]
            if len(first) < 3 or len(second) < 3:
                continue
            similarity = len(first & second) / len(first | second)
            if similarity >= threshold:
                repeated.append({
                    "first": sentences[i],
                    "second": sentences[j],
                    "similarity": round(similarity, 2),
                })

    notes = []
    if repeated:
        notes.append(
            "Some sentences repeat the same idea. Combine them, or "
            "replace one with a new point or example."
        )

    return {
        "repeated_pairs": repeated,
        "count": len(repeated),
        "level": "needs work" if repeated else "good",
        "notes": notes,
    }



EXPLANATION_MARKERS = [
    "because", "so that", "which means", "this means", "that means",
    "as a result", "that is why", "this is why", "in other words",
    "for example", "for instance", "this helps", "which helps",
    "this allows", "which allows", "due to", "therefore",
]


def check_explanation_quality(text):
    """Check whether the main ideas are explained, not just stated."""
    sentences = split_sentences(text)
    count = len(sentences)

    if count < 3:
        return {
            "marker_count": 0,
            "short_sentence_ratio": 0,
            "level": "too short to judge",
            "notes": [],
        }

    lower = text.lower()
    marker_count = 0
    for marker in EXPLANATION_MARKERS:
        pattern = r"(?<!\w)" + re.escape(marker) + r"(?!\w)"
        marker_count += len(re.findall(pattern, lower))

    short_sentences = [
        s for s in sentences
        if len(re.findall(r"\b[\w']+\b", s)) < 5
    ]
    short_ratio = round(len(short_sentences) / count, 2)

    notes = []
    if marker_count == 0:
        notes.append(
            "Explain why your points matter, using words such as "
            "'because', 'which means' or 'as a result'."
        )
    elif marker_count / count < 0.15:
        notes.append(
            "Explain more of your points instead of only stating them."
        )

    if count >= 4 and short_ratio > 0.5:
        notes.append(
            "Many of your sentences are very short. Add detail so each "
            "main idea is fully explained."
        )

    return {
        "marker_count": marker_count,
        "short_sentence_ratio": short_ratio,
        "level": "good" if not notes else "needs work",
        "notes": notes,
    }



def _stem(word):
    """Very simple stemming so 'doctors' matches 'doctor'."""
    for ending in ("ing", "ed", "es", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[:-len(ending)]
    return word


def check_completeness(text, key_points=None):
    """Check whether the speech covers the expected key points.

    key_points is a list of short phrases, for example
    ["healthcare", "education", "risks of AI"].
    """
    if not key_points:
        return {
            "checked": False,
            "covered": [],
            "missing": [],
            "coverage": None,
            "notes": [],
        }

    speech_words = {
        _stem(w) for w in re.findall(r"[a-z']+", text.lower())
    }

    covered, missing = [], []
    for point in key_points:
        words = [
            _stem(w) for w in re.findall(r"[a-z']+", point.lower())
            if w not in IDEA_STOP_WORDS
        ]
        if not words:
            continue
        matched = sum(1 for w in words if w in speech_words)
        if matched / len(words) >= 0.6:
            covered.append(point)
        else:
            missing.append(point)

    total = len(covered) + len(missing)
    coverage = round(len(covered) / total, 2) if total else None

    notes = []
    if missing:
        notes.append(
            "Your speech does not mention: " + ", ".join(missing) + "."
        )

    return {
        "checked": True,
        "covered": covered,
        "missing": missing,
        "coverage": coverage,
        "notes": notes,
    }




def _count_syllables(word):
    """Rough syllable estimate for English words."""
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)


def _reading_ease(sentence):
    """Flesch reading ease for one sentence (higher = easier)."""
    words = re.findall(r"[a-z']+", sentence.lower())
    if not words:
        return 100.0
    syllables = sum(_count_syllables(w) for w in words)
    score = 206.835 - 1.015 * len(words) - 84.6 * (syllables / len(words))
    return round(score, 1)


def analyze_readability(text, long_limit=25):
    """Judge sentences by length AND readability together.

    A long sentence is only flagged if it is also hard to read.
    """
    sentences = split_sentences(text)
    if not sentences:
        return {
            "average_reading_ease": None,
            "level": "none",
            "hard_sentences": [],
            "long_but_clear": [],
            "notes": [],
        }

    hard, long_but_clear, scores = [], [], []
    for sentence in sentences:
        word_count = len(re.findall(r"[a-z']+", sentence.lower()))
        ease = _reading_ease(sentence)
        scores.append(ease)

        if word_count > long_limit:
            if ease < 50 or word_count > 40:
                hard.append({
                    "sentence": sentence,
                    "word_count": word_count,
                    "reading_ease": ease,
                })
            else:
                long_but_clear.append({
                    "sentence": sentence,
                    "word_count": word_count,
                    "reading_ease": ease,
                })

    all_words = re.findall(r"[a-z']+", text.lower())
    total_syllables = sum(_count_syllables(w) for w in all_words)
    words_per_sentence = len(all_words) / len(sentences)
    average = round(
        206.835
        - 1.015 * words_per_sentence
        - 84.6 * (total_syllables / max(1, len(all_words))),
        1,
    )
    if average >= 60:
        level = "easy"
    elif average >= 40:
        level = "medium"
    else:
        level = "hard"

    notes = []
    if hard:
        notes.append(
            "Some long sentences are hard to follow. Split them or "
            "use simpler words."
        )
    if level == "hard" and not hard:
        notes.append(
            "Your wording is quite complex. Use simpler words where "
            "you can."
        )

    return {
        "average_reading_ease": average,
        "level": level,
        "hard_sentences": hard,
        "long_but_clear": long_but_clear,
        "notes": notes,
    }



TECHNICAL_TERMS = [
    "algorithm", "neural network", "machine learning", "deep learning",
    "dataset", "database", "encryption", "blockchain", "cloud computing",
    "big data", "natural language processing", "computer vision",
    "api", "framework", "bandwidth", "cybersecurity", "automation",
    "model training", "artificial neural", "supervised learning",
]

COMMON_ACRONYMS = {
    "AI", "TV", "OK", "USA", "UK", "US", "PC", "ID", "PM", "AM",
}

EXPLANATION_CUES = [
    "stands for", "which means", "means", "refers to", "in other words",
    "also known as", "known as", "defined as", "that is", "i.e.",
    "is a ", "is an ", "are a ", "are the ", "is the ", "is when",
    "is used to", "is how",
]


def check_technical_terms(text, extra_terms=None):
    """Check whether technical terms are explained when first used."""
    sentences = split_sentences(text)
    if not sentences:
        return {
            "terms_found": [],
            "explained": [],
            "unexplained": [],
            "notes": [],
        }

    terms = {}
    lower_sentences = [s.lower() for s in sentences]

    for term in TECHNICAL_TERMS + [t.lower() for t in (extra_terms or [])]:
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        for index, sentence in enumerate(lower_sentences):
            if re.search(pattern, sentence):
                terms[term] = index
                break

    for acronym in re.findall(r"\b[A-Z]{2,}\b", text):
        if acronym in COMMON_ACRONYMS:
            continue
        for index, sentence in enumerate(sentences):
            if re.search(r"\b" + acronym + r"\b", sentence):
                terms.setdefault(acronym, index)
                break

    explained, unexplained = [], []
    for term, index in terms.items():
        window = lower_sentences[index]
        if index + 1 < len(lower_sentences):
            window += " " + lower_sentences[index + 1]

        has_cue = any(cue in window for cue in EXPLANATION_CUES)
        has_brackets = bool(
            re.search(re.escape(term.lower()) + r"\s*\(", window)
        )
        if has_cue or has_brackets:
            explained.append(term)
        else:
            unexplained.append(term)

    notes = []
    if unexplained:
        notes.append(
            "Explain these technical terms for your audience: "
            + ", ".join(unexplained) + "."
        )

    return {
        "terms_found": list(terms.keys()),
        "explained": explained,
        "unexplained": unexplained,
        "notes": notes,
    }