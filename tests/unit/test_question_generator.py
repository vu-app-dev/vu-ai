import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.interview import Question
from services.interview.question_generator import (
    QuestionGenerator,
    _GenerateQuestionsResponse,
    _IntroResponse,
    _QuestionItem,
)

MOCK_DATA = {
    "type": "TECHNICAL",
    "difficulty": "MEDIUM",
    "technologies": ["React", "Node.js"],
    "topics": ["frontend", "APIs"],
    "estimatedTimeInMinutes": 30,
    "questions": [
        {"title": "React Hooks", "description": "Explain useEffect", "order": 1, "difficulty": "MEDIUM"},
        {"title": "REST APIs", "description": "Design a REST API", "order": 2, "difficulty": "HARD"},
    ],
}


class TestFormatExistingQuestions:
    def test_empty_questions(self):
        result = QuestionGenerator._format_existing_questions([])
        assert "None yet" in result

    def test_format_questions(self):
        qs = [
            {"title": "React Hooks", "description": "Explain useEffect", "difficulty": "MEDIUM"},
        ]
        result = QuestionGenerator._format_existing_questions(qs)
        assert "React Hooks" in result
        assert "MEDIUM" in result


class TestFallbackQuestions:
    def test_fallback_from_mock_questions(self):
        result = QuestionGenerator._fallback_questions(
            MOCK_DATA["questions"], "TECHNICAL", "MEDIUM"
        )
        assert len(result) == 2
        assert result[0].id == "q1"
        assert "React Hooks" in result[0].text
        assert result[1].id == "q2"

    def test_fallback_empty_questions(self):
        result = QuestionGenerator._fallback_questions([], "TECHNICAL", "MEDIUM")
        assert len(result) >= 1
        assert result[0].speechType == "question"

    def test_fallback_preserves_difficulty(self):
        result = QuestionGenerator._fallback_questions(
            [{"title": "Q", "description": "D", "difficulty": "HARD"}], "TECHNICAL", "MEDIUM"
        )
        assert result[0].difficulty == "HARD"


class TestQuestionCountForTime:
    def test_short_interview_gets_two_questions(self):
        assert QuestionGenerator.question_count_for_time(10) == 2

    def test_standard_interview_reserves_intro_and_closing_time(self):
        assert QuestionGenerator.question_count_for_time(30) == 6

    def test_long_interview_caps_questions(self):
        assert QuestionGenerator.question_count_for_time(60) == 10


class TestFallbackIntro:
    def test_technical_intro(self):
        intro = QuestionGenerator._fallback_intro("TECHNICAL", ["React"])
        assert "technical" in intro.lower()
        assert "React" in intro

    def test_behavioral_intro(self):
        intro = QuestionGenerator._fallback_intro("BEHAVIORAL", ["Leadership"])
        assert "experiences" in intro.lower()

    def test_coding_intro(self):
        intro = QuestionGenerator._fallback_intro("CODING", ["Algorithms"])
        assert "coding" in intro.lower()

    def test_unknown_type_defaults_to_technical(self):
        intro = QuestionGenerator._fallback_intro("UNKNOWN", [])
        assert "technical" in intro.lower()

    def test_no_technologies(self):
        intro = QuestionGenerator._fallback_intro("TECHNICAL", [])
        assert "relevant topics" in intro


class TestGenerateQuestionsWithLLM:
    @pytest.mark.asyncio
    async def test_generate_questions_llm_success(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=_GenerateQuestionsResponse(
            questions=[
                _QuestionItem(id="q1", text="What is React?", difficulty="MEDIUM", order=1),
                _QuestionItem(id="q2", text="Explain hooks", difficulty="HARD", order=2),
            ]
        ))
        gen = QuestionGenerator(llm=mock_llm)
        result = await gen.generate_questions(MOCK_DATA, cv_skills=["JavaScript"])
        assert len(result) == 2
        assert result[0].text == "What is React?"
        assert result[0].speechType == "question"

    @pytest.mark.asyncio
    async def test_generate_questions_includes_candidate_intro_context(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=_GenerateQuestionsResponse(
            questions=[
                _QuestionItem(id="q1", text="What is React?", difficulty="MEDIUM", order=1),
            ]
        ))
        gen = QuestionGenerator(llm=mock_llm)
        await gen.generate_questions(
            MOCK_DATA,
            cv_skills=["JavaScript"],
            candidate_intro="I built dashboards with React and Node.",
        )
        prompt = mock_llm.generate_json.call_args.args[0]
        assert "I built dashboards with React and Node." in prompt
        assert "topicTag" in prompt

    @pytest.mark.asyncio
    async def test_generate_questions_llm_failure_fallback(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(side_effect=Exception("LLM error"))
        gen = QuestionGenerator(llm=mock_llm)
        result = await gen.generate_questions(MOCK_DATA, cv_skills=["JavaScript"])
        assert len(result) >= 1
        assert result[0].speechType == "question"

    @pytest.mark.asyncio
    async def test_generate_questions_llm_returns_empty_fallback(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=_GenerateQuestionsResponse(questions=[]))
        gen = QuestionGenerator(llm=mock_llm)
        result = await gen.generate_questions(MOCK_DATA)
        assert len(result) >= 1


class TestGenerateIntroWithLLM:
    @pytest.mark.asyncio
    async def test_generate_intro_llm_success(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=_IntroResponse(
            intro="Welcome to your technical interview!"
        ))
        gen = QuestionGenerator(llm=mock_llm)
        result = await gen.generate_intro(
            mock_type="TECHNICAL", technologies=["React"]
        )
        assert "technical" in result.lower() or "Welcome" in result

    @pytest.mark.asyncio
    async def test_generate_intro_llm_failure_fallback(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(side_effect=Exception("LLM error"))
        gen = QuestionGenerator(llm=mock_llm)
        result = await gen.generate_intro(
            mock_type="TECHNICAL", technologies=["React"]
        )
        assert len(result) > 0
        assert "React" in result
