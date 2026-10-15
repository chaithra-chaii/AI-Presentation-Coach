import content_analysis as content

speeches = {
    "mistakes": "i think AI is a amazing tool. It dont make mistakes. The the results are more better than before. They is happy.",
    "correct": "I think AI is an amazing tool. It does not make mistakes. The results are better than before. They are happy.",
    "a university": "She is a university student and he waited an hour for a user guide.",
    "empty": "",
}

for name, text in speeches.items():
    print("---", name)
    result = content.check_grammar(text)
    print("issues:", result["issue_count"])
    for issue in result["issues"]:
        print("  ", issue["text"], "->", issue["suggestion"])