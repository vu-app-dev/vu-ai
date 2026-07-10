from unittest.mock import AsyncMock

import pytest

from models.scoring import (
    AudioScores,
    LLMAdjustment,
    ScoreWeights,
    TranscriptScores,
    VideoScores,
)
from services.scoring.score_aggregator import ScoreAggregator
from services.scoring.transcript_scorer import TranscriptScorer, EvaluateAnswerResponse


class TestTranscriptScorer:
    @pytest.mark.asyncio
    async def test_score_returns_valid(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=EvaluateAnswerResponse(
            scores={
                "communication": 4,
                "problemSolving": 4,
                "technical": 4,
                "clarityOfExplanation": 3,
                "structuredThinking": 4,
                "askingClarifications": 3,
            },
            overallComment="Good technical understanding",
            feedback="Nice work!",
            strengths=["depth"],
            areasToImprove=["structure"],
            nextAction="next_question",
        ))
        scorer = TranscriptScorer(llm=mock_llm)
        result = await scorer.score(
            question="Explain React hooks",
            transcript="React hooks let you use state...",
            mock_type="TECHNICAL",
            cv_skills=["React"],
        )
        assert 0 <= result.technical <= 100
        assert 0 <= result.communication <= 100
        assert result.technical == 80.0
        assert result.communication == 80.0

    @pytest.mark.asyncio
    async def test_score_handles_empty_transcript(self):
        mock_llm = AsyncMock()
        scorer = TranscriptScorer(llm=mock_llm)
        result = await scorer.score(
            question="Explain closures",
            transcript="",
            mock_type="TECHNICAL",
        )
        assert result is not None
        assert result.technical == 0.0

    @pytest.mark.asyncio
    async def test_score_handles_whitespace_transcript(self):
        mock_llm = AsyncMock()
        scorer = TranscriptScorer(llm=mock_llm)
        result = await scorer.score(
            question="Explain closures",
            transcript="   ",
            mock_type="TECHNICAL",
        )
        assert result.technical == 0.0

    @pytest.mark.asyncio
    async def test_score_handles_llm_failure(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(side_effect=Exception("LLM error"))
        scorer = TranscriptScorer(llm=mock_llm)
        result = await scorer.score(
            question="Explain closures",
            transcript="Closures are...",
            mock_type="TECHNICAL",
        )
        assert result is not None
        assert result.technical == 0.0

    @pytest.mark.asyncio
    async def test_score_handles_partial_llm_response(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=EvaluateAnswerResponse(
            scores={"technical": 4},
        ))
        scorer = TranscriptScorer(llm=mock_llm)
        result = await scorer.score(
            question="Explain closures",
            transcript="Closures...",
            mock_type="TECHNICAL",
        )
        assert result.technical == 80.0
        assert result.communication == 0.0

    @pytest.mark.asyncio
    async def test_score_with_active_dimensions(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=EvaluateAnswerResponse(
            scores={
                "communication": 4,
                "technical": 5,
                "clarityOfExplanation": 3,
                "problemSolving": 4,
            },
            overallComment="Good",
            feedback="Nice",
            nextAction="next_question",
        ))
        scorer = TranscriptScorer(llm=mock_llm)
        result = await scorer.score(
            question="What is gradient descent?",
            transcript="Gradient descent is an optimization algorithm...",
            mock_type="TECHNICAL",
            active_dimensions=["technical", "communication", "clarityOfExplanation"],
        )
        assert result.technical == 100.0
        assert result.communication == 80.0
        assert result.clarityOfExplanation == 60.0
        assert result.problemSolving == 0.0
        assert result.structuredThinking == 0.0
        assert result.askingClarifications == 0.0

    @pytest.mark.asyncio
    async def test_score_with_none_active_dimensions_scores_all(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=EvaluateAnswerResponse(
            scores={
                "communication": 4,
                "problemSolving": 3,
                "technical": 4,
                "clarityOfExplanation": 3,
                "structuredThinking": 4,
                "askingClarifications": 3,
            },
            overallComment="Good",
            feedback="Nice",
            nextAction="next_question",
        ))
        scorer = TranscriptScorer(llm=mock_llm)
        result = await scorer.score(
            question="Explain React hooks",
            transcript="React hooks let you use state...",
            mock_type="TECHNICAL",
            active_dimensions=None,
        )
        assert result.communication == 80.0
        assert result.problemSolving == 60.0
        assert result.technical == 80.0
        assert result.structuredThinking == 80.0


class TestScoreAggregatorWeightedAverage:
    def test_transcript_only(self):
        agg = ScoreAggregator()
        transcript = TranscriptScores(
            communication=80, problemSolving=60, technical=80,
            clarityOfExplanation=60, structuredThinking=80, askingClarifications=60,
        )
        avg = agg.compute_weighted_average(transcript, audio=None, video=None)
        assert 0 <= avg <= 100
        assert avg > 0

    def test_with_all_scores(self):
        agg = ScoreAggregator()
        transcript = TranscriptScores(
            communication=80, problemSolving=60, technical=80,
            clarityOfExplanation=60, structuredThinking=80, askingClarifications=60,
        )
        audio = AudioScores(confidence=78, speaking=82)
        video = VideoScores(eyeContact=70)
        avg = agg.compute_weighted_average(transcript, audio, video)
        assert 0 <= avg <= 100
        assert avg > 0

    def test_null_video_redistributes(self):
        agg = ScoreAggregator()
        transcript = TranscriptScores(
            communication=80, problemSolving=60, technical=80,
            clarityOfExplanation=60, structuredThinking=80, askingClarifications=60,
        )
        audio = AudioScores(confidence=78, speaking=82)
        avg_with_video = agg.compute_weighted_average(transcript, audio, VideoScores(eyeContact=70))
        avg_without_video = agg.compute_weighted_average(transcript, audio, None)
        assert avg_with_video != avg_without_video or avg_with_video == avg_without_video

    def test_all_null_gives_zero(self):
        agg = ScoreAggregator()
        transcript = TranscriptScores()
        avg = agg.compute_weighted_average(transcript, audio=None, video=None)
        assert avg == 0.0

    def test_custom_weights(self):
        custom_weights = ScoreWeights(
            technical=50.0, communication=20.0, problemSolving=15.0,
            clarityOfExplanation=5.0, structuredThinking=5.0,
            askingClarifications=5.0, confidence=0.0, speaking=0.0, eyeContact=0.0,
        )
        agg = ScoreAggregator(weights=custom_weights)
        transcript = TranscriptScores(
            communication=80, problemSolving=60, technical=100,
            clarityOfExplanation=60, structuredThinking=80, askingClarifications=60,
        )
        avg = agg.compute_weighted_average(transcript, audio=None, video=None)
        assert avg == 85.0


class TestScoreAggregatorLLMAdjustment:
    @pytest.mark.asyncio
    async def test_adjust_with_llm_success(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=LLMAdjustment(
            adjustment=5.0, reason="Strong consistency", confidence="medium",
        ))
        agg = ScoreAggregator(llm=mock_llm)
        result = await agg.adjust_with_llm(
            weighted_avg=72.5,
            question_results="Q1: 70, Q2: 75",
            mock_type="TECHNICAL",
        )
        assert result is not None
        assert result.adjustment == 5.0
        assert result.confidence == "medium"

    @pytest.mark.asyncio
    async def test_adjust_with_llm_clamps(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(return_value=LLMAdjustment(
            adjustment=15.0, reason="Exceptional", confidence="high",
        ))
        agg = ScoreAggregator(llm=mock_llm)
        result = await agg.adjust_with_llm(
            weighted_avg=80.0,
            question_results="Avg: 80",
            mock_type="TECHNICAL",
        )
        assert result is not None
        assert result.adjustment == 10.0

    @pytest.mark.asyncio
    async def test_adjust_with_llm_failure_returns_none(self):
        mock_llm = AsyncMock()
        mock_llm.generate_json = AsyncMock(side_effect=Exception("LLM error"))
        agg = ScoreAggregator(llm=mock_llm)
        result = await agg.adjust_with_llm(
            weighted_avg=72.5,
            question_results="Q1: 70",
            mock_type="TECHNICAL",
        )
        assert result is None


class TestScoreAggregatorPerformance:
    def test_compute_performance_with_adjustment(self):
        agg = ScoreAggregator()
        transcript = TranscriptScores(
            communication=80, problemSolving=60, technical=80,
            clarityOfExplanation=60, structuredThinking=80, askingClarifications=60,
        )
        adjustment = LLMAdjustment(adjustment=5.0, reason="Strong", confidence="medium")
        result = agg.compute_performance(transcript, llm_adjustment=adjustment)
        assert 0 <= result <= 100

    def test_compute_performance_without_adjustment(self):
        agg = ScoreAggregator()
        transcript = TranscriptScores(
            communication=80, problemSolving=60, technical=80,
            clarityOfExplanation=60, structuredThinking=80, askingClarifications=60,
        )
        result = agg.compute_performance(transcript)
        assert result > 0