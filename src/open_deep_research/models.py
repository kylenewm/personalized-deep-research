"""Shared Pydantic models and configurable model for the Deep Research agent."""

from typing import List, Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field as PydanticField


class FindingsReview(BaseModel):
    """Structured review of research findings for fact-checking."""

    decision: Literal["approve", "revise", "reject"] = PydanticField(
        description="approve=findings are factually grounded, revise=issues found that need fixing, reject=major fabrications detected"
    )
    confidence: float = PydanticField(
        ge=0, le=1,
        description="How confident are you in this assessment? 0-1"
    )
    issues_found: List[str] = PydanticField(
        description="List of specific issues found (fabricated names, impossible dates, uncited claims, etc.)"
    )
    suggested_fixes: List[str] = PydanticField(
        description="Specific recommendations to fix the issues"
    )
    reasoning: str = PydanticField(
        description="Overall assessment of the findings quality and factual accuracy"
    )


# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)
