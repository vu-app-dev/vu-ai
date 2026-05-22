import logging
from typing import Any

from pydantic import BaseModel, Field

from models.interview import Question
from prompts import format_prompt
from services.llm.gemini_service import GeminiService

logger = logging.getLogger(__name__)


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
    def __init__(self, gemini_service: GeminiService | None = None):
        self._gemini = gemini_service or GeminiService()

    async def generate_questions(
        self,
        mock_data: dict[str, Any],
        cv_skills: list[str] | None = None,
    ) -> list[Question]:
        mock_questions = mock_data.get("questions", [])
        mock_type = mock_data.get("type", "TECHNICAL")
        difficulty = mock_data.get("difficulty", "MEDIUM")
        technologies = mock_data.get("technologies", [])
        topics = mock_data.get("topics", [])
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

            response = await self._gemini.generate_json(
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

            response = await self._gemini.generate_json(
                prompt,
                _IntroResponse,
            )

            if response and response.intro:
                return response.intro

        except Exception as e:
            logger.warning("LLM intro generation failed, using fallback: %s", e)

        return self._fallback_intro(mock_type, technologies or [])

    @staticmethod
    def _format_existing_questions(questions: list[dict]) -> str:
        if not questions:
            return "None yet — this is the first set."
        lines = []
        for q in questions:
            title = q.get("title", "Untitled")
            desc = q.get("description", "")
            difficulty = q.get("difficulty", "MEDIUM")
            lines.append(f"- [{difficulty}] {title}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_questions(
        mock_questions: list[dict], mock_type: str, difficulty: str
    ) -> list[Question]:
        if not mock_questions:
            return [Question(
                id="q1",
                text="Tell me about your experience and what interests you about this role.",
                difficulty=difficulty,
                order=1,
                speechType="question",
            )]

        questions = []
        for i, q in enumerate(mock_questions, 1):
            title = q.get("title", "Question")
            description = q.get("description", "")
            text = f"{title}: {description}" if description else title
            questions.append(Question(
                id=f"q{i}",
                text=text,
                difficulty=q.get("difficulty", difficulty),
                order=i,
                speechType="question",
            ))
        return questions

    @staticmethod
    def _fallback_intro(mock_type: str, technologies: list[str]) -> str:
        tech_str = ", ".join(technologies) if technologies else "the relevant topics"
        intros = {
            "TECHNICAL": f"Hello! I'll be conducting your technical interview today focusing on {tech_str}. Let's get started.",
            "BEHAVIORAL": f"Hi there! I'm looking forward to learning about your experiences, especially around {tech_str}. Let's begin.",
            "CODING": f"Welcome! We'll work through some coding challenges together, focusing on {tech_str}. Ready to start?",
        }
        return intros.get(mock_type, intros["TECHNICAL"])