import sys
sys.path.insert(0, "src")

from followup_generator import generate_followup


class DemoLLM:
    def generate_json(self, prompt):
        return {
            "question": "What do you remember most about visiting your grandparents during Bihu?"
        }


profile = {
    "age": 72,
    "region": "Assam",
    "language": "English",
    "festival": "Bihu"
}

result = generate_followup(
    "Do you remember celebrating Bihu with your family?",
    "Yes, I used to visit my grandparents.",
    profile,
    llm_client=DemoLLM()
)

print(result)