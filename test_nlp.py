import sys
sys.path.insert(0, "NLP")
import nlp_analysis as n

speech = """Um, hello everyone. Today I will talk about artificial intelligence.
Basically, AI is literally changing how we work. Uh, it helps doctors,
students and businesses. In conclusion, AI is a powerful tool and we
should use it carefully."""

topic = "artificial intelligence"

print("clean:", n.clean_speech(speech))
print("fillers:", n.detect_filler_words(speech))
print("stats:", n.calculate_text_statistics(speech))
print("filler score:", n.calculate_filler_score(speech))
print("clarity score:", n.calculate_clarity_score(speech))
print("topic relevance:", n.calculate_topic_relevance(speech, topic))
print("structure:", n.analyze_speech_structure(speech))

filler = n.calculate_filler_score(speech)
clarity = n.calculate_clarity_score(speech)
topic_score = n.calculate_topic_relevance(speech, topic)
stats = n.calculate_text_statistics(speech)

overall = n.calculate_overall_score(filler, clarity, topic_score, stats["total_words"])
print("overall score:", overall)
print("feedback:", n.generate_feedback(filler, clarity, overall))


repeat_speech = """Technology is important. Technology helps students learn.
Technology helps doctors treat patients. Our future depends on technology."""

print("repetition:", n.detect_repetition(repeat_speech))

import content_analysis as c

long_speech = """Today I want to talk about how technology has changed the way that students learn in schools and universities across the world because of the internet and many new tools. For example, online classes help students in villages. Technology is useful."""

print("unclear:", c.find_unclear_sentences(long_speech))
print("examples:", c.check_supporting_examples(long_speech))
print("no examples:", c.check_supporting_examples("Technology is useful. It is good."))


import content_analysis as c

long_speech = """Today I want to talk about how technology has changed the way that students learn in schools and universities across the world because of the internet and many new tools. For example, online classes help students in villages. Technology is useful."""

print("unclear:", c.find_unclear_sentences(long_speech))
print("examples:", c.check_supporting_examples(long_speech))
print("no examples:", c.check_supporting_examples("Technology is useful. It is good."))

import scoring as s

sample = """Hello everyone. Today I will talk about artificial intelligence. Basically, AI is changing how we work. For example, doctors use AI to read scans faster and students use it to learn new topics. AI also helps businesses save time and money. In conclusion, AI is a powerful tool and we should use it carefully."""

result = s.calculate_content_score(sample, "artificial intelligence")
print("content score:", result["content_score"])
print("penalties:", result["penalties"])
print("combined with visual 70:", s.calculate_combined_score(result["content_score"], 70))
print("combined without visual:", s.calculate_combined_score(result["content_score"], None))
print("empty speech:", s.calculate_content_score("", "ai")["content_score"])