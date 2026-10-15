
import re
from collections import Counter
 
 
# 1. Filler words and expressions
# Includes single words, repeated sounds, and longer expressions.
FILLER_WORDS = [
    # Hesitation sounds
    "um", "umm", "ummm",
    "uh", "uhh", "uhhh",
    "er", "err", "errr",
    "ah", "ahh", "eh",
    "hmm", "hmmm", "mm", "mhm",
 
    # Common single-word fillers
    "actually", "basically", "literally",
    "obviously", "honestly", "seriously",
    "clearly", "definitely", "probably",
    "maybe", "perhaps", "apparently",
    "essentially", "technically", "generally",
    "certainly", "surely", "frankly",
    "personally", "interestingly",
 
    # Informal fillers
    "like", "well", "so", "anyway",
    "anyways", "right", "okay", "ok",
    "yeah", "yes", "no", "yep", "nope",
    "whatever", "somehow", "somewhat",
 
    # Short expressions
    "you know", "i mean", "i guess",
    "i think", "i suppose", "i believe",
    "kind of", "sort of", "a bit",
    "or something", "or whatever",
    "and stuff", "and things",
    "and so on", "and all that",
    "the thing is", "the thing is that",
    "what i mean", "what i mean is",
    "how do i say", "how can i say",
    "how should i say", "if you know",
    "you see", "see what i mean",
    "do you know what i mean",
    "you know what i mean",
    "if that makes sense",
    "if you understand",
    "i don't know",
 
    # Longer hesitation expressions
    "you know what",
    "what do you call it",
    "what is it called",
    "what was i saying",
    "where was i",
    "let me think",
    "let me see",
    "let's see",
    "how should i put it",
    "how can i put it",
    "how do i put it",
    "to be honest",
    "to be fair",
    "to tell you the truth",
    "at the end of the day",
    "in a way",
    "in some way",
    "in some sense",
    "for some reason",
    "as i said",
    "as i mentioned",
    "as you know",
    "as i told you",
    "the fact of the matter is",
    "what i'm trying to say",
    "what i am trying to say",
    "what i'm saying is",
    "what i am saying is",
    "the point is",
    "the point is that",
    "i would say",
    "i would like to say",
    "i just think",
    "i just feel",
    "i feel like",
    "i think that",
    "i guess that",
    "if you ask me",
    "in my opinion",
 
    # Repeated or conversational expressions
    "and then", "then again",
    "at that point",
    "basically what i mean",
    "you get me", "get what i mean",
    "if you get me", "know what i mean",
    "i mean to say",
    "what i'm saying",
    "what i want to say",
    "what i wanted to say",
    "the way i see it",
    "as a matter of fact",
    "to some extent",
    "more or less",
    "kind of like",
    "sort of like",
    "like i said",
    "like i mentioned",
    "well i mean",
    "so yeah", "yeah so",
    "okay so", "right so"
]
 
 
# 2. Words commonly ignored in clarity calculations
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "am", "be", "been", "being", "to", "of", "in",
    "on", "at", "for", "and", "or", "but", "with",
    "this", "that", "it", "i", "we", "you", "they",
    "he", "she", "my", "our", "your", "their",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should",
    "from", "as", "by", "about"
}
 
 
def clean_speech(text):
    """Normalize whitespace without removing repeated words."""
    if not text:
        return ""
 
    return re.sub(r"\s+", " ", text).strip()
 
 
def detect_filler_words(text):
    """
    Detect filler words and expressions.
 
    Longer expressions are matched first to avoid counting
    the same words multiple times.
    """
    if not text or not text.strip():
        return {}
 
    text_lower = text.lower()
    text_lower = re.sub(r"\s+", " ", text_lower)
 
    filler_counts = Counter()
    occupied = [False] * len(text_lower)
 
    # Match longer expressions before shorter expressions.
    sorted_fillers = sorted(
        set(FILLER_WORDS),
        key=len,
        reverse=True
    )
 
    for filler in sorted_fillers:
        pattern = (
            r"(?<!\w)"
            + re.escape(filler)
            + r"(?!\w)"
        )
 
        for match in re.finditer(pattern, text_lower):
            start, end = match.span()
 
            # Do not count a phrase already included
            # in a longer detected filler expression.
            if any(occupied[start:end]):
                continue
 
            # Context checks for words that are often used normally.
            if filler == "like":
                following = text_lower[end:].lstrip()
 
                # Examples: "I like music", "I like chocolate".
                # These are not automatically filler usage.
                if re.match(
                    r"(?:to\b|[a-z]+\b)",
                    following
                ):
                    next_word_match = re.match(
                        r"([a-z]+)\b", following
                    )
                    if next_word_match:
                        next_word = next_word_match.group(1)
                        common_nouns = {
                            "music", "chocolate", "books",
                            "reading", "dancing", "coding",
                            "python", "family", "rain",
                            "coffee", "tea", "food",
                            "computers", "laptops",
                            "instagram", "facebook",
                            "people", "students"
                        }
 
                        if next_word in common_nouns:
                            continue
 
            if filler == "right":
                # Avoid counting "turn right" or "right answer"
                # as a filler word.
                before = text_lower[:start].rstrip()
                after = text_lower[end:].lstrip()
 
                if re.search(r"\b(turn|go|move|correct)\s*$", before):
                    continue
 
                if re.match(r"(?:answer|side|hand|now)\b", after):
                    continue
 
            # Mark this match as counted.
            for position in range(start, end):
                occupied[position] = True
 
            filler_counts[filler] += 1
 
    return dict(filler_counts)
 
 
def calculate_filler_score(text):
    """Calculate a score based on filler-word frequency."""
    if not text or not text.strip():
        return 0
 
    words = re.findall(r"\b[\w']+\b", text)
    total_words = len(words)
 
    if total_words == 0:
        return 0
 
    filler_counts = detect_filler_words(text)
    total_fillers = sum(filler_counts.values())
 
    filler_rate = total_fillers / total_words
 
    score = 100 - (filler_rate * 200)
 
    return round(max(0, min(100, score)))
 
 
def calculate_text_statistics(text):
    """Calculate basic speech statistics."""
    if not text or not text.strip():
        return {
            "total_words": 0,
            "total_sentences": 0,
            "average_words_per_sentence": 0
        }
 
    words = re.findall(r"\b[\w']+\b", text)
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [
        sentence for sentence in sentences
        if sentence.strip()
    ]
 
    total_words = len(words)
    total_sentences = len(sentences)
 
    average_words = (
        round(total_words / total_sentences, 2)
        if total_sentences else 0
    )
 
    return {
        "total_words": total_words,
        "total_sentences": total_sentences,
        "average_words_per_sentence": average_words
    }
 
 
def calculate_clarity_score(text):
    """Estimate clarity using sentence length and repetition."""
    if not text or not text.strip():
        return 0
 
    statistics = calculate_text_statistics(text)
    total_words = statistics["total_words"]
    total_sentences = statistics["total_sentences"]
    average_words = statistics["average_words_per_sentence"]
 
    if total_words == 0:
        return 0
 
    score = 100
 
    if average_words > 25:
        score -= 20
    elif average_words > 20:
        score -= 10
    elif average_words < 3 and total_sentences > 1:
        score -= 10
 
    words = re.findall(r"\b[\w']+\b", text.lower())
 
    repeated_words = sum(
        1 for index in range(1, len(words))
        if words[index] == words[index - 1]
    )
 
    score -= min(30, repeated_words * 10)
 
    return round(max(0, min(100, score)))
 
 
def calculate_topic_relevance(speech, topic):
    """
    Estimate topic relevance for a user-provided topic.
 
    Works with topics not listed in advance, using word matching.
    """
    if not speech or not topic:
        return 0
 
    speech_words = re.findall(r"\b[a-z0-9]+\b", speech.lower())
    topic_words = re.findall(r"\b[a-z0-9]+\b", topic.lower())
 
    if not speech_words or not topic_words:
        return 0
 
    ignored_words = {
        "a", "an", "the", "is", "are", "was", "were",
        "of", "in", "on", "for", "to", "and", "or",
        "about", "my", "our", "your", "presentation",
        "topic", "importance"
    }
 
    keywords = [
        word for word in topic_words
        if word not in ignored_words
    ]
 
    if not keywords:
        keywords = topic_words
 
    unique_keywords = set(keywords)
    speech_counts = Counter(speech_words)
 
    matched_keywords = [
        word for word in unique_keywords
        if word in speech_counts
    ]
 
    if not matched_keywords:
        return 0
 
    coverage = len(matched_keywords) / len(unique_keywords)
    coverage_score = coverage * 70
 
    total_mentions = sum(
        speech_counts[word] for word in matched_keywords
    )
    mention_score = min(30, total_mentions * 5)
 
    return round(
        max(0, min(100, coverage_score + mention_score))
    )
 
 
def calculate_overall_score(
    filler_score,
    clarity_score,
    topic_relevance_score=None,
    word_count=None
):
    """Calculate the overall speech score."""
    filler_score = max(0, min(100, filler_score))
    clarity_score = max(0, min(100, clarity_score))
 
    score = filler_score * 0.40 + clarity_score * 0.60
 
    if topic_relevance_score is not None:
        topic_relevance_score = max(
            0, min(100, topic_relevance_score)
        )
        score = score * 0.70 + topic_relevance_score * 0.30
 
    if word_count is not None:
        if word_count < 10:
            score *= 0.50
        elif word_count < 30:
            score *= 0.75
 
    return round(max(0, min(100, score)))
 
 
def generate_feedback(
    filler_score,
    clarity_score,
    overall_score
):
    """Generate feedback based on speech scores."""
    feedback = []
 
    if filler_score >= 90:
        feedback.append(
            "Good job keeping filler words low."
        )
    elif filler_score >= 70:
        feedback.append(
            "Try to reduce filler words such as 'um', "
            "'basically', and 'you know'."
        )
    else:
        feedback.append(
            "Practise speaking with fewer filler words "
            "and use short pauses instead."
        )
 
    if clarity_score >= 85:
        feedback.append(
            "Your speech structure looks clear."
        )
    elif clarity_score >= 65:
        feedback.append(
            "Try using shorter and clearer sentences."
        )
    else:
        feedback.append(
            "Organize your ideas into clear sentences."
        )
 
    if overall_score >= 85:
        feedback.append(
            "Good work! Keep practising to maintain your progress."
        )
    elif overall_score >= 65:
        feedback.append(
            "You are making progress. Continue practising."
        )
    else:
        feedback.append(
            "Practise again and focus on clarity and organization."
        )
 
    return feedback
 
 
def analyze_speech_structure(text):
    """Estimate whether a speech has an introduction,
    main body, and conclusion."""
    if not text or not text.strip():
        return {
            "introduction": False,
            "main_body": False,
            "conclusion": False
        }
 
    sentences = [
        sentence.strip()
        for sentence in __import__("re").split(
            r"[.!?]+", text
        )
        if sentence.strip()
    ]
 
    words = text.lower().split()
 
    introduction_words = [
        "today", "hello", "good morning", "good afternoon",
        "firstly", "introduction", "i would like to talk"
    ]
 
    conclusion_words = [
        "finally", "in conclusion", "to conclude",
        "to sum up", "thank you", "overall"
    ]
 
    first_part = " ".join(sentences[:2]).lower()
    last_part = " ".join(sentences[-2:]).lower()
 
    has_introduction = any(
        phrase in first_part for phrase in introduction_words
    )
 
    has_conclusion = any(
        phrase in last_part for phrase in conclusion_words
    )
 
    has_main_body = len(words) >= 30 and len(sentences) >= 3
 
    return {
        "introduction": has_introduction,
        "main_body": has_main_body,
        "conclusion": has_conclusion
    }
 
 
def detect_repetition(text, min_count=3):
    """Find words and two-word phrases the speaker repeats too often."""
    words = re.findall(r"[a-z']+", text.lower())
 
    meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    word_counts = Counter(meaningful)
    repeated_words = {w: c for w, c in word_counts.items() if c >= min_count}
 
    phrases = []
    for first, second in zip(words, words[1:]):
        if first in STOP_WORDS and second in STOP_WORDS:
            continue
        phrases.append(first + " " + second)
    phrase_counts = Counter(phrases)
    repeated_phrases = {p: c for p, c in phrase_counts.items() if c >= min_count}
 
    return {
        "repeated_words": repeated_words,
        "repeated_phrases": repeated_phrases,
    }
 

