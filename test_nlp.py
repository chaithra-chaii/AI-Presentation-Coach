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