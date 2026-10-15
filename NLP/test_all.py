import nlp_analysis as nlp
import content_analysis as content

topic = "artificial intelligence"


def section(title):
    print()
    print("=" * 10, title, "=" * 10)


# ---------- 1. TOPIC RELEVANCE AND STRUCTURE ----------
section("RELEVANCE AND STRUCTURE")

speeches = {
    "good full speech": "Good morning everyone. Today I will talk about artificial intelligence. First, AI helps doctors read scans. Next, it helps students learn faster. Finally, businesses use artificial intelligence to save time. In conclusion, AI is a useful tool when we use it carefully.",
    "natural wording": "Hi everyone, my topic is artificial intelligence. It is changing healthcare because doctors can find diseases earlier. It is also changing education because students get personal help. To wrap up, the key message is that AI should support people.",
    "off topic": "Good morning everyone. Today I will talk about my favourite football team. They won the league last year. The players train every day. Thank you for listening.",
    "conclusion first": "In conclusion, AI is useful. Today I will talk about artificial intelligence. It helps doctors and students and businesses in many different ways every day.",
}

for name, text in speeches.items():
    print("---", name)
    print("relevance:", nlp.calculate_topic_relevance(text, topic))
    print("structure:", nlp.analyze_speech_structure(text))

    

# ---------- 2. TRANSITIONS ----------
section("TRANSITIONS")

transition_speeches = {
    "connected": "Today I will talk about AI. First, it helps doctors. Next, it helps students. However, it also has risks. Finally, we should use it carefully.",
    "choppy": "AI is useful. Doctors use it. Students use it. Businesses use it. It saves time.",
    "very short": "AI is useful.",
}

for name, text in transition_speeches.items():
    print("---", name)
    print(content.check_transitions(text))



# ---------- 3. LOGICAL ORDER ----------
section("LOGICAL ORDER")

order_speeches = {
    "good order": "Hello everyone. Today I will talk about AI. First, it helps doctors. Next, it helps students. Finally, it helps businesses. In conclusion, AI is a useful tool.",
    "conclusion too early": "In conclusion, AI is useful. Today I will talk about artificial intelligence. It helps doctors and students and businesses.",
    "introduction too late": "AI helps doctors. AI helps students. AI helps businesses. AI saves time. Hello everyone, I will talk about AI.",
    "points out of order": "Hello everyone. Finally, AI saves time. First, AI helps doctors. Next, AI helps students. Thank you.",
    "very short": "AI is useful.",
}

for name, text in order_speeches.items():
    print("---", name)
    print(content.check_logical_order(text))




# ---------- 4. REPEATED IDEAS ----------
section("REPEATED IDEAS")

idea_speeches = {
    "repeated idea": "Hello everyone. AI helps doctors read scans. Doctors use AI to read scans. AI helps students learn faster. Thank you.",
    "all different": "AI helps doctors read scans. It helps students learn faster. Businesses use it to save money.",
    "empty": "",
}

for name, text in idea_speeches.items():
    print("---", name)
    print(content.check_repeated_ideas(text))




# ---------- 5. EXPLANATION QUALITY ----------
section("EXPLANATION QUALITY")

explanation_speeches = {
    "well explained": "AI helps doctors because it can read scans faster than people. This means patients get answers sooner. AI also helps students, for example by giving personal practice questions. As a result, learning becomes easier.",
    "bare statements": "AI is useful. Doctors use it. Students use it. Businesses use it. It saves time.",
    "very short": "AI is useful.",
}

for name, text in explanation_speeches.items():
    print("---", name)
    print(content.check_explanation_quality(text))

    

# ---------- 6. COMPLETENESS ----------
section("COMPLETENESS")

key_points = ["healthcare", "education", "risks of AI"]

completeness_speeches = {
    "covers all": "AI helps healthcare because doctors find diseases earlier. In education, students get personal help. However, there are risks of AI, such as wrong answers.",
    "misses one": "AI helps doctors in healthcare. It also supports education for students.",
    "covers none": "I like football. My team won last year.",
    "no key points given": "AI helps doctors in healthcare.",
}

for name, text in completeness_speeches.items():
    print("---", name)
    if name == "no key points given":
        print(content.check_completeness(text))
    else:
        print(content.check_completeness(text, key_points))

        

# ---------- 7. FULL RESULT ----------
section("FULL RESULT")

import json
import feedback_generator as fg

full_speech = "Hello everyone. Today I will talk about artificial intelligence. First, AI helps healthcare because doctors find diseases earlier. Next, in education, students get personal help. However, AI can make mistakes. In conclusion, AI is a useful tool."

result = fg.build_analysis_result(
    full_speech,
    "artificial intelligence",
    key_points=["healthcare", "education", "risks of AI"],
)
print("strengths:", json.dumps(result["strengths"], indent=2))
print("areas_to_improve:", json.dumps(result["areas_to_improve"], indent=2))
print("suggestions:", json.dumps(result["suggestions"], indent=2))
print("details keys:", list(result["details"].keys()))
print("empty speech:", fg.build_analysis_result("", "AI"))



# ---------- 8. READABILITY ----------
section("READABILITY")

readability_speeches = {
    "long but easy": "AI helps doctors and it helps students and it helps teachers and it helps farmers and it helps shops and it helps banks and it helps schools every day.",
    "long and hard": "Contemporary organizations increasingly leverage sophisticated computational methodologies to optimize organizational decision-making and operational efficiency across heterogeneous infrastructure environments while simultaneously maintaining comprehensive regulatory compliance throughout international jurisdictions.",
    "short and simple": "AI helps doctors. It helps students. It saves time.",
    "empty": "",
}

for name, text in readability_speeches.items():
    print("---", name)
    print(content.analyze_readability(text))

    

# ---------- 9. TECHNICAL TERMS ----------
section("TECHNICAL TERMS")

term_speeches = {
    "explained": "Machine learning means computers learn from data. NLP stands for natural language processing. It helps AI understand text.",
    "unexplained": "We use machine learning and NLP with an algorithm. The neural network reads the dataset.",
    "no technical terms": "AI helps doctors. It helps students. It saves time.",
    "empty": "",
}

for name, text in term_speeches.items():
    print("---", name)
    print(content.check_technical_terms(text))

print("full speech readability:", content.analyze_readability(full_speech)["average_reading_ease"])




# ---------- 10. TOPIC RELEVANCE (IMPROVED) ----------
section("TOPIC RELEVANCE (IMPROVED)")

relevance_topic = "cybersecurity"

relevance_speeches = {
    "uses the word": "Cybersecurity protects computers and networks from digital attacks. Strong password and multi-factor authentication help protect accounts.",
    "without the word": "Hackers send phishing emails to steal passwords. A firewall and encryption protect your network. Always update your software.",
    "mixed": "Cybersecurity protects networks from attacks. My favourite football team won the league. Strong passwords protect accounts. The players train every day.",
    "off topic": "My favourite football team won the league. The players train every day.",
    "empty": "",
}

for name, text in relevance_speeches.items():
    print("---", name)
    print(content.analyze_topic_relevance(text, relevance_topic))



# ---------- 11. UNKNOWN TOPIC ----------
section("UNKNOWN TOPIC")

space_speech = "Rockets carry astronauts to orbit. The rocket launches from a pad. Engineers test every part before the mission."

print("--- unknown topic, no extra keywords")
print(content.analyze_topic_relevance(space_speech, "space exploration"))

print("--- unknown topic, with extra keywords")
print(content.analyze_topic_relevance(
    space_speech, "space exploration",
    extra_terms=["rocket", "astronaut", "orbit", "mission"],
))



# ---------- 12. TOPIC IN FULL RESULT ----------
section("TOPIC IN FULL RESULT")

mixed_speech = "Hello everyone. Today I will talk about cybersecurity. Strong passwords protect accounts because attackers guess weak ones. My favourite football team won the league. In conclusion, use strong passwords."

topic_result = fg.build_analysis_result(mixed_speech, "cybersecurity")
print("areas_to_improve:", topic_result["areas_to_improve"])
print("topic_analysis score:", topic_result["details"]["topic_analysis"]["score"])
print("confidence:", topic_result["details"]["topic_analysis"]["confidence"])



# ---------- 13. PRACTICE PLAN ----------
section("PRACTICE PLAN")

strong_speech = "Hello everyone. Today I will talk about cybersecurity. First, strong passwords protect accounts because attackers guess weak ones. For example, a long phrase is harder to guess. Next, multi-factor authentication adds a second check. Finally, updating software closes known security holes. In conclusion, these three habits keep your accounts safe."
weak_speech = "Cybersecurity is important. Hackers are bad. Um, passwords. Uh, basically yes."

print("--- strong speech")
print(fg.build_analysis_result(strong_speech, "cybersecurity")["practice_next"])
print("--- weak speech")
print(fg.build_analysis_result(weak_speech, "cybersecurity")["practice_next"])
print("--- empty speech")
print(fg.build_analysis_result("", "cybersecurity")["practice_next"])  




# ---------- 14. REPETITION SCORE ----------
section("REPETITION SCORE")

repetitive_speech = "AI helps doctors read scans. Doctors use AI to read scans. AI helps students learn. AI helps teachers teach. AI helps businesses grow. AI saves time."
varied_speech = "Hello everyone. Today I will talk about artificial intelligence. It helps doctors read scans faster. Students get personal practice. Businesses save money. In conclusion, AI is a useful tool."

for name, text in {"repetitive": repetitive_speech, "varied": varied_speech, "empty": ""}.items():
    result = fg.build_analysis_result(text, "artificial intelligence")
    print(name, "-> repetition_score:", result["repetition_score"])