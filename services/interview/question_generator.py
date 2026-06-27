import logging
from typing import Any

from pydantic import BaseModel, Field

from models.interview import Question
from prompts import format_prompt
from services.llm.llm_service import LLMService

logger = logging.getLogger(__name__)


def _normalize_str_list(items: list) -> list[str]:
    """Convert a list that may contain dicts to a list of strings."""
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            result.append(item.get("title", item.get("name", item.get("text", str(item)))))
        else:
            result.append(str(item))
    return result


class _QuestionItem(BaseModel):
    id: str
    text: str
    difficulty: str = "MEDIUM"
    order: int = 1


class _GenerateQuestionsResponse(BaseModel):
    questions: list[_QuestionItem] = Field(default_factory=list)


class _IntroResponse(BaseModel):
    intro: str = ""


class QuestionGenerator:
    def __init__(self, llm: LLMService | None = None):
        self._llm = llm or LLMService()

    async def generate_questions(
        self,
        mock_data: dict[str, Any],
        cv_skills: list[str] | None = None,
    ) -> list[Question]:
        mock_questions = mock_data.get("questions", [])
        mock_type = mock_data.get("type", "TECHNICAL")
        difficulty = mock_data.get("difficulty", "MEDIUM")
        technologies = _normalize_str_list(mock_data.get("technologies", []))
        topics = _normalize_str_list(mock_data.get("topics", []))
        estimated_time = mock_data.get("estimatedTimeInMinutes", 30)

        try:
            num_questions = max(3, min(8, estimated_time // 5))
            existing = self._format_existing_questions(mock_questions)

            prompt = format_prompt(
                "generate_questions",
                mock_type=mock_type,
                difficulty=difficulty,
                technologies=technologies,
                topics=topics,
                estimated_time_minutes=estimated_time,
                num_questions=num_questions,
                cv_skills=cv_skills or [],
                existing_questions=existing,
                cv_skills_section="CV Skills: " + ", ".join(cv_skills) if cv_skills else "",
            )

            response = await self._llm.generate_json(
                prompt,
                _GenerateQuestionsResponse,
            )

            if response and response.questions:
                return [
                    Question(
                        id=q.id,
                        text=q.text,
                        difficulty=q.difficulty,
                        order=q.order,
                        speechType="question",
                    )
                    for q in response.questions
                ]

        except Exception as e:
            logger.warning("LLM question generation failed, using fallback: %s", e)

        return self._fallback_questions(mock_questions, mock_type, difficulty)

    async def generate_intro(
        self,
        mock_type: str,
        technologies: list[str] | None = None,
        topics: list[str] | None = None,
        estimated_time: int = 30,
        difficulty: str = "MEDIUM",
        cv_skills: list[str] | None = None,
        cv_summary: str = "",
    ) -> str:
        try:
            prompt = format_prompt(
                "interview_intro",
                mock_type=mock_type,
                difficulty=difficulty,
                technologies=technologies or [],
                topics=topics or [],
                estimated_time_minutes=estimated_time,
                cv_skills=cv_skills or [],
                cv_summary=cv_summary,
            )

            response = await self._llm.generate_json(
                prompt,
                _IntroResponse,
            )

            if response and response.intro:
                return response.intro

        except Exception as e:
            logger.warning("LLM intro generation failed, using fallback: %s", e)

        return self._fallback_intro(mock_type, technologies or [])

    @staticmethod
    def _format_existing_questions(questions: list) -> str:
        if not questions:
            return "None yet — this is the first set."
        lines = []
        for q in questions:
            if isinstance(q, str):
                lines.append(f"- {q}")
            elif isinstance(q, dict):
                title = q.get("title", "Untitled")
                desc = q.get("description", "")
                difficulty = q.get("difficulty", "MEDIUM")
                lines.append(f"- [{difficulty}] {title}: {desc}")
            else:
                lines.append(f"- {str(q)}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_questions(
        mock_questions: list[dict], mock_type: str, difficulty: str
    ) -> list[Question]:
        difficulty = difficulty.upper() if difficulty else "MEDIUM"
        mock_type = mock_type.upper() if mock_type else "TECHNICAL"

        if mock_questions:
            questions = []
            for i, q in enumerate(mock_questions, 1):
                title = q.get("title", "Question") if isinstance(q, dict) else str(q)
                description = q.get("description", "") if isinstance(q, dict) else ""
                text = f"{title}: {description}" if description else title
                q_difficulty = ((q.get("difficulty") if isinstance(q, dict) else None) or difficulty).upper()
                questions.append(Question(
                    id=f"q{i}",
                    text=text,
                    difficulty=q_difficulty,
                    order=i,
                    speechType="question",
                ))
            return questions

        # Generate generic questions based on type
        templates = {
            "TECHNICAL": [
                "Tell me about your experience and what interests you about this role.",
                "Can you walk me through a challenging technical problem you've solved?",
                "How do you approach debugging a complex issue in production?",
                "Explain a technical concept you're passionate about as if teaching a beginner.",
                "What trade-offs do you consider when choosing between different technologies?",
            ],
            "BEHAVIORAL": [
                "Tell me about a time you had to work with a difficult team member.",
                "Describe a situation where you had to adapt to a significant change.",
                "How do you handle tight deadlines and competing priorities?",
                "Can you give an example of when you took initiative beyond your role?",
                "Tell me about a failure and what you learned from it.",
            ],
            "CODING": [
                "How would you approach designing a system that needs to handle millions of requests?",
                "Explain the difference between time complexity and space complexity with an example.",
                "How would you optimize a slow database query?",
                "Describe how you would implement a caching strategy.",
                "What's your approach to writing testable code?",
            ],
        }
        defaults = [
            "Tell me about your experience and what interests you about this role.",
            "Can you describe a challenging project you've worked on?",
            "Where do you see yourself growing in this field?",
            "What's something you've learned recently that excited you?",
            "How do you stay current with developments in your field?",
        ]
        question_texts = templates.get(mock_type, defaults)

        return [
            Question(id=f"q{i}", text=text, difficulty=difficulty, order=i, speechType="question")
            for i, text in enumerate(question_texts, 1)
        ]

    @staticmethod
    def _fallback_intro(mock_type: str, technologies: list[str]) -> str:
        mock_type = mock_type.upper() if mock_type else "TECHNICAL"
        tech_str = ", ".join(technologies) if technologies else "the relevant topics"
        intros = {
            "TECHNICAL": f"Hello! I'll be conducting your technical interview today focusing on {tech_str}. Let's get started.",
            "BEHAVIORAL": f"Hi there! I'm looking forward to learning about your experiences, especially around {tech_str}. Let's begin.",
            "CODING": f"Welcome! We'll work through some coding challenges together, focusing on {tech_str}. Ready to start?",
        }
        return intros.get(mock_type, intros["TECHNICAL"])