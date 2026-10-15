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