import sys, json
sys.path.insert(0, "NLP")
import feedback_generator as f

sample = """Hello everyone. Today I will talk about artificial intelligence. Basically, AI is changing how we work. AI helps doctors. AI helps students. AI helps businesses. AI is useful."""

result = f.build_analysis_result(sample, "artificial intelligence", visual_score=70)

print("content score:", result["content_score"])
print("combined score:", result["combined_score"])
print(json.dumps({
    "strengths": result["strengths"],
    "areas_to_improve": result["areas_to_improve"],
    "suggestions": result["suggestions"],
}, indent=2))