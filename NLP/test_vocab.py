import content_analysis as content

speeches = {
    "informal": "Guys, we are gonna talk about AI today. It does lots of cool stuff and it is awesome.",
    "advanced": "Contemporary organizations increasingly leverage sophisticated computational methodologies to optimize organizational decision-making and operational efficiency.",
    "simple": "AI is a tool. It can help us do work. We use it to see and to talk.",
    "empty": "",
}

for name, text in speeches.items():
    print("---", name)
    print(content.analyze_vocabulary(text, "general"))