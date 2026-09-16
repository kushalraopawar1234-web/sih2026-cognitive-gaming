import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from followup_generator import generate_followup
from llm_generator import InvalidLLMResponseError
from scenario_generator import CulturalDataError, generate_scenario


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate_json(self, prompt):
        self.prompts.append(prompt)
        return self.response


class ScenarioTests(unittest.TestCase):
    def profile(self, region, festival):
        return {
            "age": 72,
            "region": region,
            "language": "English",
            "festival": festival,
            "favourite_food": "home cooking",
            "childhood_activity": "visiting relatives",
            "preferred_topic": "family memories",
        }

    def test_supported_festival_scenarios(self):
        examples = [
            ("Assam", "Bihu"),
            ("Meghalaya", "Wangala"),
            ("Manipur", "Yaoshang"),
            ("Mizoram", "Chapchar Kut"),
            ("Nagaland", "Hornbill Festival"),
        ]
        for region, festival in examples:
            with self.subTest(region=region):
                fake = FakeLLM({
                    "title": festival + " memories",
                    "scenario": "A short, gentle invitation to share a memory.",
                    "question": "What would you like to share, if you feel comfortable?",
                    "follow_up_topics": ["family", "food"],
                })
                result = generate_scenario(self.profile(region, festival), llm_client=fake)
                self.assertEqual(result["title"], festival + " memories")
                self.assertIn(festival, fake.prompts[0])
                self.assertIn(region, fake.prompts[0])

    def test_missing_cultural_information_is_rejected(self):
        fake = FakeLLM({})
        with self.assertRaises(CulturalDataError):
            generate_scenario(self.profile("Assam", "Unknown Festival"), llm_client=fake)
        self.assertEqual(fake.prompts, [])

    def test_unsupported_language_is_rejected(self):
        fake = FakeLLM({})
        profile = self.profile("Assam", "Bihu")
        profile["language"] = "Unsupported language"
        with self.assertRaises(CulturalDataError):
            generate_scenario(profile, llm_client=fake)
        self.assertEqual(fake.prompts, [])

    def test_invalid_scenario_response_is_rejected(self):
        fake = FakeLLM({"title": "Only a title"})
        with self.assertRaises(InvalidLLMResponseError):
            generate_scenario(self.profile("Assam", "Bihu"), llm_client=fake)

    def test_followup_returns_one_question(self):
        fake = FakeLLM({"question": "What do you remember most about visiting your grandparents?"})
        result = generate_followup(
            "Do you remember celebrating Bihu with your family?",
            "Yes, I used to visit my grandparents.",
            self.profile("Assam", "Bihu"),
            llm_client=fake,
        )
        self.assertEqual(result["question"], "What do you remember most about visiting your grandparents?")
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
