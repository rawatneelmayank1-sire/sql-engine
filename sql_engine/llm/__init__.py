"""The verified LLM text-to-SQL front-end."""

from __future__ import annotations

from .client import (
    DEFAULT_GROQ_MODEL,
    FixtureMissingError,
    GroqClient,
    LLMClient,
    LLMError,
    Prompt,
    RecordingClient,
    ReplayClient,
    StaticClient,
)
from .pipeline import Answer, Attempt, TextToSQL, extract_sql
from .prompts import SYSTEM_PROMPT, build_repair_message, build_user_message
from .verifier import (
    Failure,
    FailureCategory,
    Repair,
    VerificationResult,
    Verifier,
    estimated_scan_bytes,
)

__all__ = [
    "Answer",
    "Attempt",
    "DEFAULT_GROQ_MODEL",
    "Failure",
    "FailureCategory",
    "FixtureMissingError",
    "GroqClient",
    "LLMClient",
    "LLMError",
    "Prompt",
    "RecordingClient",
    "Repair",
    "ReplayClient",
    "SYSTEM_PROMPT",
    "StaticClient",
    "TextToSQL",
    "VerificationResult",
    "Verifier",
    "build_repair_message",
    "build_user_message",
    "estimated_scan_bytes",
    "extract_sql",
]
