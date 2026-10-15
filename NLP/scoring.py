import nlp_analysis as nlp
import content_analysis as content


def calculate_content_score(text, topic=None):
    """Content score out of 100, with the details behind it."""
    if not text or not text.strip():
        return {"content_score": 0, "details": {}, "penalties": []}

    stats = nlp.calculate_text_statistics(text)
    filler = nlp.calculate_filler_score(text)
    clarity = nlp.calculate_clarity_score(text)
    relevance = nlp.calculate_topic_relevance(text, topic) if topic else None
    structure = nlp.analyze_speech_structure(text)
    repetition = nlp.detect_repetition(text)
    unclear = content.find_unclear_sentences(text)
    examples = content.check_supporting_examples(text)

    score = nlp.calculate_overall_score(
        filler, clarity, relevance, stats["total_words"]
    )

    penalties = []
    for part in ["introduction", "main_body", "conclusion"]:
        if not structure[part]:
            score -= 5
            penalties.append("Missing " + part.replace("_", " "))
    if not examples["has_examples"]:
        score -= 5
        penalties.append("No supporting examples")
    if unclear:
        score -= min(9, len(unclear) * 3)
        penalties.append("Long or unclear sentences")
    if repetition["repeated_words"]:
        score -= min(6, len(repetition["repeated_words"]) * 2)
        penalties.append("Repeated words")

    return {
        "content_score": round(max(0, min(100, score))),
        "details": {
            "filler_score": filler,
            "clarity_score": clarity,
            "topic_relevance": relevance,
            "structure": structure,
            "repetition": repetition,
            "unclear_sentences": unclear,
            "examples": examples,
        },
        "penalties": penalties,
    }


def calculate_combined_score(content_score, visual_score=None):
    """Combine content and visual scores. Missing visual data is handled."""
    if content_score is None and visual_score is None:
        return None
    if visual_score is None:
        return round(content_score)
    if content_score is None:
        return round(visual_score)
    return round(content_score * 0.70 + visual_score * 0.30)