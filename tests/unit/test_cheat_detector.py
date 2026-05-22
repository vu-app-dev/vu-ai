import pytest

from models.interview import CheatClassification, CheatEvidence
from services.interview.cheat_detector import CheatDetector


class TestCheatDetectorTabSwitches:
    def test_clean_0_tabs(self):
        result = CheatDetector().classify(tab_count=0)
        assert result.level == "Clean"
        assert result.evidence.tabSwitches == 0

    def test_clean_1_tab(self):
        result = CheatDetector().classify(tab_count=1)
        assert result.level == "Clean"

    def test_clean_2_tabs(self):
        result = CheatDetector().classify(tab_count=2)
        assert result.level == "Clean"
        assert result.evidence.tabSwitches == 2

    def test_flagged_3_tabs(self):
        result = CheatDetector().classify(tab_count=3)
        assert result.level == "Flagged"
        assert result.evidence.tabSwitches == 3

    def test_flagged_5_tabs(self):
        result = CheatDetector().classify(tab_count=5)
        assert result.level == "Flagged"

    def test_critical_6_tabs(self):
        result = CheatDetector().classify(tab_count=6)
        assert result.level == "Critical"

    def test_critical_10_tabs(self):
        result = CheatDetector().classify(tab_count=10)
        assert result.level == "Critical"

    def test_default_no_tabs(self):
        result = CheatDetector().classify()
        assert result.level == "Clean"
        assert result.evidence.tabSwitches == 0


class TestCheatDetectorVideoFlags:
    def test_no_face_pct_flagged(self):
        result = CheatDetector().classify(tab_count=0, no_face_pct=25)
        assert result.level == "Flagged"

    def test_no_face_pct_critical(self):
        result = CheatDetector().classify(tab_count=0, no_face_pct=50)
        assert result.level == "Critical"

    def test_multiple_face_pct_flagged(self):
        result = CheatDetector().classify(tab_count=0, multiple_face_pct=15)
        assert result.level == "Flagged"

    def test_multiple_face_pct_critical(self):
        result = CheatDetector().classify(tab_count=0, multiple_face_pct=25)
        assert result.level == "Critical"

    def test_gaze_away_pct_flagged(self):
        result = CheatDetector().classify(tab_count=0, gaze_away_pct=45)
        assert result.level == "Flagged"

    def test_gaze_away_pct_critical(self):
        result = CheatDetector().classify(tab_count=0, gaze_away_pct=85)
        assert result.level == "Critical"

    def test_below_thresholds_stays_clean(self):
        result = CheatDetector().classify(
            tab_count=0, no_face_pct=10, multiple_face_pct=5, gaze_away_pct=20
        )
        assert result.level == "Clean"

    def test_combined_flags_escalate_to_critical(self):
        result = CheatDetector().classify(tab_count=3, no_face_pct=25)
        assert result.level == "Critical"


class TestCheatDetectorEvidence:
    def test_evidence_populated(self):
        result = CheatDetector().classify(
            tab_count=4, no_face_pct=30, multiple_face_pct=12, gaze_away_pct=50
        )
        assert result.evidence.tabSwitches == 4
        assert result.evidence.noFacePct == 30
        assert result.evidence.multipleFacePct == 12
        assert result.evidence.gazeAwayPct == 50

    def test_evidence_null_video_by_default(self):
        result = CheatDetector().classify(tab_count=2)
        assert result.evidence.noFacePct is None
        assert result.evidence.multipleFacePct is None
        assert result.evidence.gazeAwayPct is None