import json
import feedback_generator as fg

speech = """Hello guys. Today I will talk about AI. i think AI is a amazing tool. It does lots of cool stuff. For example, doctors use AI to read scans. In conclusion, AI helps everyone."""

result = fg.build_analysis_result(speech, "AI")
print(json.dumps(result, indent=2))

print("--- empty speech")
print(fg.build_analysis_result("", "AI"))