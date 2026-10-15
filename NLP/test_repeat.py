import nlp_analysis as nlp

speeches = {
    "repeated AI": "AI is useful. AI helps students. AI helps doctors. AI helps businesses.",
    "repeated phrase": "In my opinion, this is good. In my opinion, it is fast. In my opinion, we should use it.",
    "no repetition": "Today I will explain artificial intelligence. It supports healthcare, education and business in different ways.",
}

for name, text in speeches.items():
    print(name, "->", nlp.detect_repetition(text))