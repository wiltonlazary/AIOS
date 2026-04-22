"""
Memory content formatter for the AIOS personalization pipeline.

Converts JSON memory content into natural language sentences based
on the ``memory_type`` metadata field.  Operates as a pure function
at inject time — stored memory content is never modified.

Supported memory types:
  - ``"profile"``      → user profile sentence
  - ``"task_context"`` → task context sentence
  - ``"conversation"`` → returned as-is
  - unknown / missing  → returned as-is
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# -- Profile formatting templates ----------------------------------

_PROFILE_TEMPLATES: dict[str, str] = {
    "name": "Their name is {value}.",
    "language": "They prefer coding in {value}.",
    "tools": "They like using {value}.",
    "style": "They prefer a {value} response style.",
}

_PROFILE_KEY_ORDER: list[str] = [
    "name",
    "language",
    "tools",
    "style",
]

# -- Task-context formatting templates -----------------------------

_TASK_TEMPLATES: dict[str, str] = {
    "project": "Working on project {value}.",
    "experiment": "Running experiment: {value}.",
    "goals": "Goals: {value}.",
    "blockers": "Blockers: {value}.",
    "next_steps": "Next steps: {value}.",
}

_TASK_KEY_ORDER: list[str] = [
    "project",
    "experiment",
    "goals",
    "blockers",
    "next_steps",
]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _try_parse_json(content: str) -> Optional[dict]:
    """Attempt to parse *content* as a JSON object.

    Returns the parsed ``dict`` on success, or ``None`` when
    *content* is not valid JSON or is not a JSON object.
    """
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _value_to_str(value: object) -> str:
    """Convert a value to its string representation.

    Lists are joined with ``", "``.  All other types are
    converted via ``str()``.
    """
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _format_profile(data: dict) -> str:
    """Format a profile JSON dict into a natural language
    sentence prefixed with ``"User profile:"``."""
    parts: list[str] = ["User profile:"]

    for key in _PROFILE_KEY_ORDER:
        if key in data:
            value = _value_to_str(data[key])
            parts.append(
                _PROFILE_TEMPLATES[key].format(
                    value=value
                )
            )

    for key, value in data.items():
        if key not in _PROFILE_TEMPLATES:
            parts.append(
                f"{key}: {_value_to_str(value)}."
            )

    return " ".join(parts)


def _format_task_context(data: dict) -> str:
    """Format a task-context JSON dict into a natural language
    sentence prefixed with ``"Current task context:"``."""
    parts: list[str] = ["Current task context:"]

    for key in _TASK_KEY_ORDER:
        if key in data:
            value = _value_to_str(data[key])
            parts.append(
                _TASK_TEMPLATES[key].format(value=value)
            )

    for key, value in data.items():
        if key not in _TASK_TEMPLATES:
            parts.append(
                f"{key}: {_value_to_str(value)}."
            )

    return " ".join(parts)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def format_memory(content: str, metadata: dict) -> str:
    """Convert memory content to natural language based on
    ``memory_type``.

    Args:
        content: Raw memory content string (may be JSON or
            plain text).
        metadata: Memory metadata dict containing
            ``memory_type``.

    Returns:
        A natural language string.  Falls back to the original
        *content* on any error or unrecognised memory type.
    """
    try:
        memory_type = metadata.get("memory_type", "")
        if not memory_type:
            return content

        if memory_type == "conversation":
            return content

        if memory_type not in ("profile", "task_context"):
            return content

        data = _try_parse_json(content)
        if data is None:
            return content

        if memory_type == "profile":
            return _format_profile(data)

        return _format_task_context(data)
    except Exception:
        logger.warning(
            "Memory formatting failed, returning raw "
            "content",
            exc_info=True,
        )
        return content
