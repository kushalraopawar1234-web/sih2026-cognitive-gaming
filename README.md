# sih2026-cognitive-gaming

# AI Cultural Narrative Module

This is the AI and Cultural Narrative Engineer module for SIH26003. It creates short, respectful reminiscence scenarios for elderly users, grounded in a human-reviewable North Eastern Region (NER) cultural dataset. It is only the narrative module, not the complete hackathon application.

## My role

The AI & Cultural Narrative Engineer:

- integrates and prompt-engineers the LLM;
- turns reviewed cultural context into gentle reminiscence scenarios;
- personalizes prompts from an elderly user's profile;
- generates one adaptive follow-up question;
- keeps the provider replaceable for the backend team; and
- adds cultural grounding and safety checks.

## Architecture

```text
backend API
    |
    +--> scenario_generator.generate_scenario(profile)
    |        +--> ner_culture.json (verified-style context)
    |        +--> reminiscence_prompt.txt
    |        +--> llm_generator.LLMClient
    |
    +--> followup_generator.generate_followup(question, response, profile)
             +--> LLMClient
```

- `data/ner_culture.json`: state-by-state festivals, foods, activities, languages, themes, and review metadata. The entries are deliberately high-level and must be checked by a human cultural reviewer before production.
- `prompts/reminiscence_prompt.txt`: reusable scenario prompt with profile fields, grounding instructions, safety rules, and a JSON output contract.
- `src/llm_generator.py`: standard-library HTTP client for an OpenAI-compatible provider. It reads environment variables and exposes errors that a backend can map to HTTP responses.
- `src/scenario_generator.py`: validates the profile, rejects unsupported region/festival combinations, fills the prompt, and validates the structured result.
- `src/followup_generator.py`: asks for and validates exactly one follow-up question.
- `tests/test_scenarios.py`: offline tests using a fake client; no API key or network call is needed.

## Dataset review and safety

The dataset is a conservative starting point, not proof that every detail fits every community. A teammate should review entries with official publications and local community/cultural reviewers. The generator rejects a festival that is not present for the selected region and tells the model to use only the supplied context.

Every prompt requires simple language, one question at a time, permission to skip, no pressure, no correction of a user's memory, no frightening content, a short interaction, and no medical diagnosis. A real product should also provide a human support/escalation path outside this module.

## LLM configuration

The default client expects an OpenAI-compatible chat-completions endpoint. Never put the key in source code.

PowerShell:

```powershell
$env:LLM_API_KEY = "your-key"
$env:LLM_API_URL = "https://api.openai.com/v1/chat/completions"
$env:LLM_MODEL = "gpt-4o-mini"
$env:LLM_PROVIDER = "openai-compatible"
```

`LLM_PROVIDER` currently accepts `openai-compatible`. The `LLMClient` constructor can also be replaced with a teammate's provider adapter or a fake test client. This keeps provider-specific code in one file.

## Install and run in VS Code

Python 3.9 or newer is recommended. No third-party package is required.

```powershell
cd AI-Cultural-Narrative
python -m unittest discover -s tests -v
```

To use the real provider, configure the environment variables above, then call the function from a Python backend. The module can be imported by adding `src` to the backend's package path, or by packaging `src` according to the backend team's existing structure.

## Example input

```python
from scenario_generator import generate_scenario

profile = {
    "age": 72,
    "region": "Assam",
    "language": "English",
    "festival": "Bihu",
    "favourite_food": "pitha",
    "childhood_activity": "visiting grandparents",
    "preferred_topic": "family memories",
}

result = generate_scenario(profile)
print(result)
```

Expected shape:

```json
{
  "title": "Bihu family memories",
  "scenario": "Bihu can bring thoughts of family, food, and spring. You may skip this question if you prefer.",
  "question": "What do you remember about being with family during Bihu?",
  "follow_up_topics": ["grandparents", "pitha", "music"]
}
```

The exact wording is produced by the configured LLM and may differ. The code validates the required fields before returning them.

## Follow-up example

```python
from followup_generator import generate_followup

followup = generate_followup(
    "Do you remember celebrating Bihu with your family?",
    "Yes, I used to visit my grandparents.",
    profile,
)
```

Expected shape:

```json
{"question": "What do you remember most about visiting your grandparents during Bihu?"}
```

## Errors a backend can handle

- `MissingAPIKeyError`: configure `LLM_API_KEY`.
- `APIRequestError`: provider/network failure; retry or return a temporary-service response.
- `InvalidLLMResponseError`: provider did not return the required JSON; log safely and retry/fallback.
- `CulturalDataError`: region or festival is absent from the reviewed dataset.
- `ValueError`: required profile or exchange fields are missing.

Do not log sensitive patient responses unnecessarily. Apply the backend's authentication, consent, retention, and access-control policies around this module.
