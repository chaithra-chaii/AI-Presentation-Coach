import nlp_analysis as nlp
import content_analysis as content

speeches = {
    "short clear": "AI helps doctors. It helps students. It helps businesses.",
    "one long sentence": "Today I want to talk about how artificial intelligence has changed the way that students learn in schools and universities across the world because of the internet and many new tools and also how it helps doctors and businesses every single day.",
    "exactly 5 words": "AI is useful for everyone.",
    "question and exclamation": "Is AI useful? Yes! It helps everyone.",
}

for name, text in speeches.items():
    print("---", name)
    print("stats:", nlp.calculate_text_statistics(text))
    print("clarity:", nlp.calculate_clarity_score(text))
    print("unclear:", content.find_unclear_sentences(text))