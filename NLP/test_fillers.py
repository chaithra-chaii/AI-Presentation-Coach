import nlp_analysis as nlp

speeches = {
    "obvious fillers": "Um, today I will, uh, talk about AI. Basically, you know, it is like really useful.",
    "clean speech": "Today I will explain artificial intelligence. It helps doctors diagnose disease and helps students learn faster.",
    "normal words": "So we built a model. Well, the results were good. No errors appeared, and yes, the team was happy.",
    "normal 'like'": "I like music and I like coffee. Students like to learn with new tools.",
}

for name, text in speeches.items():
    fillers = nlp.detect_filler_words(text)
    print(name, "->", fillers)
    print("   score:", nlp.calculate_filler_score(text))