import pytest

from models.scoring import AudioScores
from services.scoring.audio_scorer import AudioScorer


class TestAudioScorerWPM:
    def test_wpm_normal(self):
        scorer = AudioScorer()
        wpm = scorer._words_per_minute(150, 120)
        assert 70 <= wpm <= 80

    def test_wpm_fast(self):
        scorer = AudioScorer()
        wpm = scorer._words_per_minute(300, 60)
        assert wpm == 300

    def test_wpm_zero_duration(self):
        scorer = AudioScorer()
        wpm = scorer._words_per_minute(100, 0)
        assert wpm == 0.0


class TestAudioScorerScore:
    def test_fluent_speaker(self):
        result = AudioScorer().score(word_count=150, duration_seconds=120, filler_count=3, pause_count=5)
        assert 0 <= result.confidence <= 100
        assert 0 <= result.speaking <= 100
        assert result.speaking > 50
        assert result.confidence > 20

    def test_silent(self):
        result = AudioScorer().score(word_count=0, duration_seconds=120)
        assert result.confidence < 30
        assert result.speaking < 20

    def test_fast_talker(self):
        result = AudioScorer().score(word_count=400, duration_seconds=60, filler_count=1)
        assert result.confidence > 10
        assert result.speaking > 60

    def test_very_slow_talker(self):
        result = AudioScorer().score(word_count=20, duration_seconds=120)
        assert result.confidence < 50

    def test_many_fillers(self):
        result = AudioScorer().score(word_count=150, duration_seconds=120, filler_count=20)
        assert result.confidence < 80

    def test_many_pauses(self):
        result = AudioScorer().score(word_count=150, duration_seconds=120, pause_count=50)
        assert result.speaking < 80

    def test_zero_duration(self):
        result = AudioScorer().score(word_count=100, duration_seconds=0)
        assert result.confidence == 0.0
        assert result.speaking == 0.0

    def test_reasonable_answer(self):
        result = AudioScorer().score(word_count=120, duration_seconds=90, filler_count=5, pause_count=8)
        assert 30 <= result.confidence <= 100
        assert 50 <= result.speaking <= 100


class TestCountFillers:
    def test_basic_fillers(self):
        transcript = "Um, I think that, uh, like, I would say basically"
        count = AudioScorer.count_fillers(transcript)
        assert count >= 4

    def test_no_fillers(self):
        transcript = "I have experience with React and Node.js"
        count = AudioScorer.count_fillers(transcript)
        assert count == 0

    def test_empty_transcript(self):
        count = AudioScorer.count_fillers("")
        assert count == 0

    def test_multi_word_fillers(self):
        transcript = "You know, I think sort of like this is kind of what I mean"
        count = AudioScorer.count_fillers(transcript)
        assert count >= 3

    def test_filler_case_insensitive(self):
        transcript = "Um, UH, LIKE, Basically"
        count = AudioScorer.count_fillers(transcript)
        assert count >= 3