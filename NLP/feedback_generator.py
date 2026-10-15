import scoring
import content_analysis


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

def build_analysis_result(text, topic=None, visual_score=None,
                          audience="general", key_points=None):
    """Main function for Member 4: returns the full structured result."""
    analysis = scoring.calculate_content_score(text, topic)
    feedback = generate_detailed_feedback(analysis)
    details = dict(analysis["details"]) if analysis["details"] else {}

    if text and text.strip():
        vocabulary = content_analysis.analyze_vocabulary(text, audience)
        grammar = content_analysis.check_grammar(text)
        transitions = content_analysis.check_transitions(text)
        order = content_analysis.check_logical_order(text)
        ideas = content_analysis.check_repeated_ideas(text)
        explanation = content_analysis.check_explanation_quality(text)
        completeness = content_analysis.check_completeness(text, key_points)
        readability = content_analysis.analyze_readability(text)
        terms = content_analysis.check_technical_terms(text)

        if vocabulary["notes"]:
            feedback["areas_to_improve"].append(
                "Some word choices could be improved."
            )
            feedback["suggestions"].extend(vocabulary["notes"])

        for issue in grammar["issues"][:3]:
            feedback["areas_to_improve"].append(
                "Grammar: '" + issue["text"] + "'"
            )
            feedback["suggestions"].append(issue["suggestion"])

        if transitions["level"] == "good":
            feedback["strengths"].append(
                "You connect your ideas with transition phrases."
            )
        elif transitions["notes"]:
            feedback["areas_to_improve"].append(
                "Your ideas could be connected more smoothly."
            )
            feedback["suggestions"].extend(transitions["notes"])

        if order["level"] == "good":
            feedback["strengths"].append(
                "Your ideas are presented in a sensible order."
            )
        for issue in order["issues"]:
            feedback["areas_to_improve"].append("Order of ideas.")
            feedback["suggestions"].append(issue)

        if ideas["notes"]:
            feedback["areas_to_improve"].append(
                "Some sentences repeat the same idea."
            )
            feedback["suggestions"].extend(ideas["notes"])

        if explanation["level"] == "good":
            feedback["strengths"].append(
                "You explain your main ideas, not just state them."
            )
        elif explanation["notes"]:
            feedback["areas_to_improve"].append(
                "Your main ideas need more explanation."
            )
            feedback["suggestions"].extend(explanation["notes"])

        if completeness["checked"]:
            if not completeness["missing"]:
                feedback["strengths"].append(
                    "You cover all the key points."
                )
            else:
                feedback["areas_to_improve"].append(
                    "Some key points are missing."
                )
                feedback["suggestions"].extend(completeness["notes"])

        if readability["level"] == "easy" and not readability["hard_sentences"]:
            feedback["strengths"].append(
                "Your language is easy to read and understand."
            )
        elif readability["notes"]:
            feedback["areas_to_improve"].append(
                "Some sentences are hard to read."
            )
            feedback["suggestions"].extend(readability["notes"])

        if terms["unexplained"]:
            feedback["areas_to_improve"].append(
                "Some technical terms are not explained."
            )
            feedback["suggestions"].extend(terms["notes"])
        elif terms["explained"]:
            feedback["strengths"].append(
                "You explain your technical terms."
            )

        details["vocabulary"] = vocabulary
        details["grammar"] = grammar
        details["transitions"] = transitions
        details["logical_order"] = order
        details["repeated_ideas"] = ideas
        details["explanation_quality"] = explanation
        details["completeness"] = completeness
        details["readability"] = readability
        details["technical_terms"] = terms

    return {
        "content_score": analysis["content_score"],
        "combined_score": scoring.calculate_combined_score(
            analysis["content_score"], visual_score
        ),
        "strengths": feedback["strengths"],
        "areas_to_improve": feedback["areas_to_improve"],
        "suggestions": feedback["suggestions"],
        "details": details,
    }