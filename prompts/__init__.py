import re
from pathlib import Path

from prompts.bars_anchors import get_bars_anchors
from prompts.persona import get_persona_instruction, MOCK_TYPE_TECHNICAL, VALID_MOCK_TYPES

PROMPTS_DIR = Path(__file__).parent
PERSONA_DELIMITER_OPEN = "<candidate_content>"
PERSONA_DELIMITER_CLOSE = "</candidate_content>"


def _normalize_str_list(items: list) -> list[str]:
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            result.append(item.get("title", item.get("name", item.get("text", str(item)))))
        else:
            result.append(str(item))
    return result


def format_prompt(template_name: str, **kwargs) -> str:
    if "mock_type" in kwargs:
        mock_type = kwargs["mock_type"]
        if mock_type not in VALID_MOCK_TYPES:
            mock_type = MOCK_TYPE_TECHNICAL
        kwargs["persona_instruction"] = get_persona_instruction(mock_type)
        active_dims = kwargs.pop("active_dimensions", None)
        kwargs["bars_anchors"] = get_bars_anchors(mock_type, active_dimensions=active_dims)
        if active_dims is not None:
            dim_labels = ", ".join(active_dims)
            kwargs["active_dimensions"] = (
                f"ACTIVE DIMENSIONS FOR THIS QUESTION: {dim_labels}\n"
                f"Only score these dimensions. For any dimension NOT listed above, return score 0."
            )
        else:
            kwargs["active_dimensions"] = "Score ALL four dimensions for this question."
        kwargs["mock_type"] = mock_type

    if "cv_skills" in kwargs and isinstance(kwargs["cv_skills"], list):
        kwargs["cv_skills"] = ", ".join(_normalize_str_list(kwargs["cv_skills"]))

    if "technologies" in kwargs and isinstance(kwargs["technologies"], list):
        kwargs["technologies"] = ", ".join(_normalize_str_list(kwargs["technologies"]))

    if "topics" in kwargs and isinstance(kwargs["topics"], list):
        kwargs["topics"] = ", ".join(_normalize_str_list(kwargs["topics"]))

    if "existing_questions" not in kwargs:
        kwargs["existing_questions"] = "None yet — this is the first set."

    if "question_number" not in kwargs:
        kwargs["question_number"] = 1

    if "total_questions" not in kwargs:
        kwargs["total_questions"] = "5-8"

    if "mock_number" not in kwargs:
        kwargs["mock_number"] = 1

    if "total_mocks" not in kwargs:
        kwargs["total_mocks"] = 1

    if "asked_questions" not in kwargs:
        kwargs["asked_questions"] = "None yet — this is the first question."

    if "conversation_history" not in kwargs:
        kwargs["conversation_history"] = "None — this is the first question."

    if "cv_skills_section" not in kwargs:
        cv_skills = kwargs.get("cv_skills", "")
        if cv_skills:
            kwargs["cv_skills_section"] = f"CV Skills: {cv_skills}"
        else:
            kwargs["cv_skills_section"] = ""

    template_file = PROMPTS_DIR / f"{template_name}.txt"
    if not template_file.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_file}")

    template = template_file.read_text(encoding="utf-8")

    for placeholder in _extract_placeholders(template):
        if placeholder not in kwargs:
            kwargs[placeholder] = ""

    return template.format(**kwargs)


def _extract_placeholders(template: str) -> set[str]:
    return set(re.findall(r'\{(\w+)\}', template))