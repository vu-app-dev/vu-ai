import logging
from dataclasses import dataclass, field

from models.scoring import VideoScores

logger = logging.getLogger(__name__)


@dataclass
class VideoFrameResult:
    face_detected: bool = False
    num_faces: int = 0
    eye_contact: float | None = None
    gaze_horizontal: float | None = None


class VideoScorer:
    def compute_session_scores(self, frame_results: list[VideoFrameResult]) -> VideoScores:
        if not frame_results:
            return VideoScores(eyeContact=None)

        total = len(frame_results)
        valid_eye = [r for r in frame_results if r.eye_contact is not None]

        if not valid_eye:
            return VideoScores(eyeContact=None)

        avg_eye_contact = sum(r.eye_contact for r in valid_eye) / len(valid_eye)
        return VideoScores(eyeContact=round(avg_eye_contact, 1))

    def compute_cheat_metrics(self, frame_results: list[VideoFrameResult]) -> dict[str, float | None]:
        total = len(frame_results)
        if total == 0:
            return {"noFacePct": None, "multipleFacePct": None, "gazeAwayPct": None}

        no_face = sum(1 for r in frame_results if not r.face_detected)
        multiple_faces = sum(1 for r in frame_results if r.num_faces > 1)

        valid_gaze = [r for r in frame_results if r.gaze_horizontal is not None and r.face_detected]
        gaze_away = sum(1 for r in valid_gaze if abs(r.gaze_horizontal) > 0.3)

        return {
            "noFacePct": round(no_face / total * 100, 1),
            "multipleFacePct": round(multiple_faces / total * 100, 1),
            "gazeAwayPct": round(gaze_away / len(valid_gaze) * 100, 1) if valid_gaze else None,
        }