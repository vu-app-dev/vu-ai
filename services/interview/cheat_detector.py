from models.interview import CheatClassification, CheatEvidence


class CheatDetector:
    TAB_THRESHOLD_FLAGGED = 3
    TAB_THRESHOLD_CRITICAL = 6

    def classify(
        self,
        tab_count: int = 0,
        no_face_pct: float | None = None,
        multiple_face_pct: float | None = None,
        gaze_away_pct: float | None = None,
        speaker_count: int | None = None,
        second_speaker_pct: float | None = None,
    ) -> CheatClassification:
        signals: list[str] = []
        evidence = CheatEvidence(
            tabSwitches=tab_count,
            noFacePct=no_face_pct,
            multipleFacePct=multiple_face_pct,
            gazeAwayPct=gaze_away_pct,
            speakerCount=speaker_count,
            secondSpeakerPct=second_speaker_pct,
        )

        if tab_count >= self.TAB_THRESHOLD_CRITICAL:
            signals.append(f"Excessive tab switches ({tab_count})")
        elif tab_count >= self.TAB_THRESHOLD_FLAGGED:
            signals.append(f"Multiple tab switches ({tab_count})")

        if no_face_pct is not None and no_face_pct > 20:
            signals.append(f"Face not detected {no_face_pct:.0f}% of frames")
        if multiple_face_pct is not None and multiple_face_pct > 10:
            signals.append(f"Multiple faces detected {multiple_face_pct:.0f}% of frames")
        if gaze_away_pct is not None and gaze_away_pct > 40:
            signals.append(f"Gaze away {gaze_away_pct:.0f}% of frames")

        if speaker_count is not None and speaker_count >= 2 and second_speaker_pct is not None:
            if second_speaker_pct > 15:
                signals.append(f"Multiple speakers detected ({speaker_count}, second speaker {second_speaker_pct:.0f}%)")
            elif second_speaker_pct > 5:
                signals.append(f"Second speaker detected ({second_speaker_pct:.0f}%)")

        level = self._determine_level(
            tab_count, no_face_pct, multiple_face_pct, gaze_away_pct,
            speaker_count, second_speaker_pct,
        )

        return CheatClassification(
            level=level,
            evidence=evidence,
        )

    @staticmethod
    def _determine_level(
        tab_count: int = 0,
        no_face_pct: float | None = None,
        multiple_face_pct: float | None = None,
        gaze_away_pct: float | None = None,
        speaker_count: int | None = None,
        second_speaker_pct: float | None = None,
    ) -> str:
        critical_count = 0
        flagged_count = 0

        if tab_count >= 6:
            critical_count += 1
        elif tab_count >= 3:
            flagged_count += 1

        if no_face_pct is not None:
            if no_face_pct > 40:
                critical_count += 1
            elif no_face_pct > 20:
                flagged_count += 1

        if multiple_face_pct is not None:
            if multiple_face_pct > 20:
                critical_count += 1
            elif multiple_face_pct > 10:
                flagged_count += 1

        if gaze_away_pct is not None:
            if gaze_away_pct > 80:
                critical_count += 1
            elif gaze_away_pct > 40:
                flagged_count += 1

        if speaker_count is not None and speaker_count >= 2 and second_speaker_pct is not None:
            if second_speaker_pct > 15:
                critical_count += 1
            elif second_speaker_pct > 5:
                flagged_count += 1

        if critical_count >= 1:
            return "Critical"
        if flagged_count >= 2:
            return "Critical"
        if flagged_count >= 1:
            return "Flagged"
        return "Clean"