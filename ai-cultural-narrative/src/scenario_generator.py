"""Generate culturally grounded reminiscence scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from llm_generator import LLMClient, InvalidLLMResponseError, LLMError


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "ner_culture.json"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "reminiscence_prompt.txt"
REQUIRED_PROFILE_FIELDS = (
    "age",
    "region",
    "language",
    "festival",
    "favourite_food",
    "childhood_activity",
    "preferred_topic",
)
REQUIRED_SCENARIO_FIELDS = ("title", "scenario", "question", "follow_up_topics")


class CulturalDataError(ValueError):
    """Raised when requested cultural data is absent or unsupported."""


def _load_dataset() -> Dict[str, Any]:
    try:
        return json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CulturalDataError("The cultural dataset is unavailable or invalid.") from exc


def _validate_profile(patient_profile: Dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_PROFILE_FIELDS if not patient_profile.get(field)]
    if missing:
        raise ValueError(f"Patient profile is missing: {', '.join(missing)}")


def _get_cultural_context(region: str, festival: str) -> Dict[str, Any]:
    dataset = _load_dataset()
    region_data = dataset.get("regions", {}).get(region)
    if not region_data:
        raise CulturalDataError(f"Unsupported or missing region: {region}")
    festival_data = next(
        (item for item in region_data.get("festivals", []) if item["name"].casefold() == festival.casefold()),
        None,
    )
    if not festival_data:
        raise CulturalDataError(f"Festival '{festival}' is not verified for {region}.")
    return {"region": region, "festival": festival_data, **region_data}


def _validate_language(language: str, cultural_context: Dict[str, Any]) -> None:
    supported_languages = cultural_context.get("languages", [])
    if not any(language.casefold() == item.casefold() for item in supported_languages):
        raise CulturalDataError(f"Unsupported language for this cultural context: {language}")


def _validate_scenario(result: Dict[str, Any]) -> Dict[str, Any]:
    missing = [field for field in REQUIRED_SCENARIO_FIELDS if field not in result]
    if missing:
        raise InvalidLLMResponseError(f"Scenario JSON is missing: {', '.join(missing)}")
    if not isinstance(result["follow_up_topics"], list):
        raise InvalidLLMResponseError("follow_up_topics must be a list.")
    return result


def generate_scenario(
    patient_profile: Dict[str, Any],
    cultural_context: Optional[Dict[str, Any]] = None,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Return one structured scenario, ready to pass to a backend response."""
    _validate_profile(patient_profile)
    verified_context = cultural_context or _get_cultural_context(
        patient_profile["region"], patient_profile["festival"]
    )
    _validate_language(patient_profile["language"], verified_context)
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        **patient_profile,
        cultural_context=json.dumps(verified_context, ensure_ascii=False, indent=2),
    )
    result = (llm_client or LLMClient()).generate_json(prompt)
    return _validate_scenario(result)
