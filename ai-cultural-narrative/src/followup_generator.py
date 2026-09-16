"""Generate one gentle follow-up question from the previous exchange."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from llm_generator import LLMClient, InvalidLLMResponseError


FOLLOWUP_TEMPLATE = """You are a gentle reminiscence companion for an older adult.

Previous question: {previous_question}
Patient response: {patient_response}
Patient profile: {patient_profile}

Ask exactly ONE short, respectful follow-up question based only on the response.
Use simple language. Never pressure the person to remember. Never say their memory is wrong.
The person may skip. Avoid frightening or distressing content. Do not diagnose or give medical advice.
Return only valid JSON in this exact shape:
{{"question": "one simple question"}}
"""


def generate_followup(
    previous_question: str,
    patient_response: str,
    patient_profile: Dict[str, Any],
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, str]:
    """Return exactly one validated follow-up question."""
    if not previous_question.strip() or not patient_response.strip():
        raise ValueError("previous_question and patient_response are required.")
    if not patient_profile:
        raise ValueError("patient_profile is required.")

    prompt = FOLLOWUP_TEMPLATE.format(
        previous_question=previous_question,
        patient_response=patient_response,
        patient_profile=json.dumps(patient_profile, ensure_ascii=False),
    )
    result = (llm_client or LLMClient()).generate_json(prompt)
    question = result.get("question") if isinstance(result, dict) else None
    if not isinstance(question, str) or not question.strip():
        raise InvalidLLMResponseError("Follow-up JSON must contain one non-empty question.")
    return {"question": question.strip()}
