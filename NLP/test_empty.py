import nlp_analysis as nlp
import content_analysis as content
import feedback_generator as fg

for text in ["", "   "]:
    print("--- testing:", repr(text))
    tests = {
        "fillers": lambda: nlp.detect_filler_words(text),
        "stats": lambda: nlp.calculate_text_statistics(text),
        "clarity": lambda: nlp.calculate_clarity_score(text),
        "structure": lambda: nlp.analyze_speech_structure(text),
        "repetition": lambda: nlp.detect_repetition(text),
        "unclear": lambda: content.find_unclear_sentences(text),
        "examples": lambda: content.check_supporting_examples(text),
        "full result": lambda: fg.build_analysis_result(text),
    }
    for name, run in tests.items():
        try:
            print(name, "->", run())
        except Exception as error:
            print(name, "-> CRASHED:", error)