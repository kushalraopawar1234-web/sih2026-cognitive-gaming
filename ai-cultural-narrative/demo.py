import sys
sys.path.insert(0, "src")

from scenario_generator import generate_scenario


class DemoLLM:
    def generate_json(self, prompt):
        return {
            "title": "Bihu Family Memories",
            "scenario": "Bihu can bring memories of family, food, and spring.",
            "question": "What do you remember about celebrating Bihu?",
            "follow_up_topics": ["family", "pitha", "music"]
        }


profile = {
    "age": 72,
    "region": "Assam",
    "language": "English",
    "festival": "Bihu",
    "favourite_food": "pitha",
    "childhood_activity": "visiting grandparents",
    "preferred_topic": "family memories"
}

result = generate_scenario(profile, llm_client=DemoLLM())
print(result)