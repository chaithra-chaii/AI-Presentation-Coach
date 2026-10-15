import scoring


def generate_detailed_feedback(analysis):
    """Turn the scoring details into strengths, areas to improve and suggestions."""
    d = analysis["details"]
    strengths, improve, suggestions = [], [], []

    if not d:
        return {
            "strengths": [],
            "areas_to_improve": ["No speech was detected."],
            "suggestions": ["Record your presentation again and speak clearly."],
        }

    # Structure
    structure = d["structure"]
    if all(structure.values()):
        strengths.append("Your speech has a clear introduction, main body and conclusion.")
    if not structure["introduction"]:
        improve.append("The speech has no clear introduction.")
        suggestions.append("Start by greeting the audience and stating your topic.")
    if not structure["main_body"]:
        improve.append("The main body needs more detail.")
        suggestions.append("Add two or three main points and explain each one.")
    if not structure["conclusion"]:
        improve.append("The speech has no clear conclusion.")
        suggestions.append("Summarize your main points at the end, then thank the audience.")

    # Filler words
    filler = d["filler_score"]
    if filler >= 90:
        strengths.append("You kept filler words very low.")
    elif filler >= 70:
        improve.append("Some filler words appear in your speech.")
        suggestions.append("Replace words like 'um' and 'basically' with a short pause.")
    else:
        improve.append("Too many filler words.")
        suggestions.append("Practise slowly and pause silently instead of using filler words.")

    # Clarity
    if d["clarity_score"] >= 85:
        strengths.append("Your sentences are clear and easy to follow.")

    # Topic relevance
    relevance = d["topic_relevance"]
    if relevance is not None:
        if relevance >= 70:
            strengths.append("Your speech stays on topic.")
        elif relevance < 50:
            improve.append("The speech does not stay close to the topic.")
            suggestions.append("Mention your main topic words more often and link each point to them.")

    # Examples
    if d["examples"]["has_examples"]:
        strengths.append("You support your points with examples or facts.")
    else:
        improve.append("Your points are not supported by examples.")
        suggestions.append("Add a practical example, for example: 'For instance, ...'.")

    # Long sentences
    for item in d["unclear_sentences"][:3]:
        start = " ".join(item["sentence"].split()[:8])
        improve.append("A sentence is too long: '" + start + "...'")
        suggestions.append("Split the sentence starting '" + start + "...' into two shorter sentences.")

    # Repetition
    for word, count in list(d["repetition"]["repeated_words"].items())[:3]:
        improve.append("You repeated '" + word + "' " + str(count) + " times.")
        suggestions.append("Use a different word or a new example instead of repeating '" + word + "'.")

    return {
        "strengths": strengths,
        "areas_to_improve": improve,
        "suggestions": suggestions,
    }


def build_analysis_result(text, topic=None, visual_score=None):
    """Main function for Member 4: returns the full structured result."""
    analysis = scoring.calculate_content_score(text, topic)
    feedback = generate_detailed_feedback(analysis)

    return {
        "content_score": analysis["content_score"],
        "combined_score": scoring.calculate_combined_score(
            analysis["content_score"], visual_score
        ),
        "strengths": feedback["strengths"],
        "areas_to_improve": feedback["areas_to_improve"],
        "suggestions": feedback["suggestions"],
        "details": analysis["details"],
    }