import math

import re

from models.scoring import AudioScores

FILLER_WORDS = frozenset({
    "um", "uh", "like", "you know", "kind of", "sort of",
    "basically", "actually", "literally", "so yeah",
    "i mean", "right", "okay so", "well",
})

SLOW_WPM = 60
FAST_WPM = 180
IDEAL_WPM = 120
MAX_FILLER_PER_MINUTE = 20


class AudioScorer:
    def score(
        self,
        word_count: int,
        duration_seconds: float,
        filler_count: int = 0,
        pause_count: int = 0,
    ) -> AudioScores:
        if duration_seconds <= 0 or word_count == 0:
            return AudioScores(confidence=0.0, speaking=0.0)

        wpm = self._words_per_minute(word_count, duration_seconds)
        confidence = self._confidence_score(wpm, filler_count, duration_seconds)
        speaking = self._speaking_score(wpm, pause_count, duration_seconds)

        return AudioScores(
            confidence=round(confidence, 1),
            speaking=round(speaking, 1),
        )

    @staticmethod
    def _words_per_minute(word_count: int, duration_seconds: float) -> float:
        if duration_seconds <= 0:
            return 0.0
        return (word_count / duration_seconds) * 60

    @staticmethod
    def _confidence_score(wpm: float, filler_count: int, duration_seconds: float) -> float:
        wpm_score = 0.0
        if SLOW_WPM <= wpm <= FAST_WPM:
            ideal_dist = abs(wpm - IDEAL_WPM)
            max_dist = max(IDEAL_WPM - SLOW_WPM, FAST_WPM - IDEAL_WPM)
            wpm_score = max(0.0, 100 - (ideal_dist / max_dist) * 25)
        elif wpm > FAST_WPM:
            wpm_score = max(30.0, 75 - (wpm - FAST_WPM) * 1.5)
        else:
            wpm_score = max(20.0, wpm / SLOW_WPM * 60)

        duration_minutes = duration_seconds / 60
        filler_rate = filler_count / max(duration_minutes, 0.1)
        filler_penalty = min(filler_rate / MAX_FILLER_PER_MINUTE, 1.0) * 15

        return max(0.0, min(100.0, wpm_score - filler_penalty))

    @staticmethod
    def _speaking_score(wpm: float, pause_count: int, duration_seconds: float) -> float:
        speaking_time_estimate = (wpm / 60) * duration_seconds
        if duration_seconds <= 0:
            return 0.0

        speaking_ratio = min(speaking_time_estimate / duration_seconds, 1.0)

        pause_rate_penalty = 0.0
        if duration_seconds > 0:
            pause_ratio = pause_count * 2 / duration_seconds
            pause_rate_penalty = min(pause_ratio, 1.0) * 30

        base_score = speaking_ratio * 80
        score = max(0.0, min(100.0, base_score + 20 - pause_rate_penalty))

        return score

    @staticmethod
    def count_fillers(transcript: str) -> int:
        if not transcript:
            return 0
        lower = transcript.lower()
        lower = re.sub(r'[^\w\s]', '', lower)
        count = 0
        for filler in FILLER_WORDS:
            if " " in filler:
                count += lower.count(filler)
            else:
                words = lower.split()
                count += sum(1 for w in words if w == filler)
        return count