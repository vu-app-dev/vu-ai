# Plan: Comprehensive Cheat Detection

## Overview

Detect if the interviewee is cheating during the interview. The system uses **5 signals** across 3 modalities (browser, video, audio) to classify the candidate as **Clean**, **Flagged**, or **Critical**.

## Cheating Scenarios & How We Detect Them

| Scenario | How candidate cheats | Detection signal | Status |
|---|---|---|---|
| Looking up answers | Switches tabs to Google/ChatGPT | **Tab switches** | DONE |
| Someone else at screen | Another person sits at the computer | **Multiple faces detected** | DONE |
| Leaving the camera | Steps away, phone reads answers | **No face detected** | DONE |
| Reading from another screen/notes | Eyes look away from camera | **Gaze away** | DONE |
| Someone feeds answers verbally | Helper whispers/dictates, candidate repeats | **Speaker diarization** | TO IMPLEMENT |

---

## Signal 1: Tab Switches (DONE)

**What it catches**: Candidate switches to another tab (Google, ChatGPT, notes).

**How it works**: Frontend tracks `visibilitychange` events, sends `tab_switch` WebSocket messages with `totalCount`. Stored on `Session.tabSwitches`.

**Thresholds**:
| Count | Level |
|---|---|
| 0-2 | Clean |
| 3-5 | Flagged |
| 6+ | Critical |

**Files**: `routers/interview.py` (receives WS message), `session_manager.py` (stores count), `cheat_detector.py` (classifies)

---

## Signal 2: No Face Detected (DONE)

**What it catches**: Candidate leaves camera view — possibly stepping away to read answers from a phone or another screen, or someone else taking over without being on camera.

**How it works**: Frontend sends video frames via `video_frame` WebSocket messages. MediaPipe Face Mesh processes each frame. `VideoScorer.compute_cheat_metrics()` calculates percentage of frames with no face detected.

**Thresholds**:
| % of frames | Level |
|---|---|
| 0-20% | Clean |
| 20-40% | Flagged |
| 40%+ | Critical |

**Files**: `routers/interview.py` (receives frames), `services/scoring/video_scorer.py` (computes metrics), `cheat_detector.py` (classifies)

---

## Signal 3: Multiple Faces Detected (DONE)

**What it catches**: Someone else is visible on camera — could be looking over shoulder, sitting next to the candidate, or someone helping from behind.

**How it works**: MediaPipe counts faces per frame (`num_faces`). `VideoScorer.compute_cheat_metrics()` calculates percentage of frames with >1 face.

**Thresholds**:
| % of frames | Level |
|---|---|
| 0-10% | Clean |
| 10-20% | Flagged |
| 20%+ | Critical |

**Files**: `services/scoring/video_scorer.py`, `cheat_detector.py`

---

## Signal 4: Gaze Away (DONE)

**What it catches**: Candidate frequently looking away from screen — reading notes, a second monitor, or someone's hand signals.

**How it works**: MediaPipe tracks horizontal gaze direction. If `abs(gaze_horizontal) > 0.3`, the candidate is looking away. Percentage calculated over frames where face is detected and gaze is measured.

**Thresholds**:
| % of frames | Level |
|---|---|
| 0-40% | Clean |
| 40-80% | Flagged |
| 80%+ | Critical |

**Files**: `services/scoring/video_scorer.py`, `cheat_detector.py`

---

## Signal 5: Speaker Diarization — TO IMPLEMENT

**What it catches**: Someone off-camera feeds answers to the candidate verbally (whispering, dictating). The candidate repeats the answers so they speak naturally — tab switches and face detection can't catch this.

**How it works**: Buffer raw audio during the interview. At `end_session`, send to **AssemblyAI batch API** with `speaker_labels=True`. The API returns utterances tagged by speaker (`"A"`, `"B"`, etc.). If >1 speaker detected, compute the percentage of audio from non-primary speakers.

**Why batch API**: AssemblyAI streaming v3 (`TurnEvent`) has **no speaker field**. Speaker diarization requires full audio context to cluster voices — it's inherently a post-processing step. Only the batch API supports `speaker_labels=True` (returns `Utterance` objects with `speaker` field).

**Thresholds**:
| Condition | Level |
|---|---|
| 1 speaker | Clean |
| 2+ speakers, second speaker < 5% | Clean (background noise) |
| 2+ speakers, second speaker 5-15% | Flagged |
| 2+ speakers, second speaker > 15% | Critical |
| Diarization failed/unavailable | No signal (null) — no effect |

---

## Classification Logic (DONE — needs update for speaker signal)

**File**: `services/interview/cheat_detector.py`

Each signal independently contributes a `flagged_count` or `critical_count`:

```
if ANY signal → Critical:     critical_count += 1
elif ANY signal → Flagged:    flagged_count += 1
```

**Final level**:
```
critical_count >= 1           → Critical
flagged_count >= 2            → Critical   (two separate Flagged signals = Critical)
flagged_count == 1            → Flagged
else                          → Clean
```

This means: a candidate Flagged on tabs AND Flagged on gaze = **Critical** (combined signals escalate).

---

## Implementation Steps (Speaker Diarization only)

Everything below is NEW code. Signals 1-4 are already fully implemented and tested.

### Step 1: Add speaker evidence fields to `CheatEvidence`

**File:** `models/interview.py`

Add two optional fields to `CheatEvidence`:

```python
class CheatEvidence(BaseModel):
    tabSwitches: int = 0
    noFacePct: Optional[float] = None
    multipleFacePct: Optional[float] = None
    gazeAwayPct: Optional[float] = None
    speakerCount: Optional[int] = None          # NEW
    secondSpeakerPct: Optional[float] = None    # NEW
```

### Step 2: Create `SpeakerAnalyzer` service

**File:** `services/scoring/speaker_analyzer.py` (NEW FILE)

Takes buffered audio, sends to AssemblyAI batch API with `speaker_labels=True`, returns speaker count + second speaker percentage.

```python
import logging
import tempfile
import os
from dataclasses import dataclass

import assemblyai as aai
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SpeakerAnalysis:
    speaker_count: int
    second_speaker_pct: float
    utterances_by_speaker: dict[str, float]  # speaker label → total duration in seconds


class SpeakerAnalyzer:
    def __init__(self):
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY

    async def analyze(self, audio_bytes: bytes, sample_rate: int = 16000) -> SpeakerAnalysis | None:
        """
        Run speaker diarization on raw PCM audio.
        Returns SpeakerAnalysis or None if diarization fails.
        """
        if not audio_bytes or len(audio_bytes) < sample_rate * 2:  # less than 1 second
            logger.warning("[Speaker] Audio too short for diarization")
            return None

        tmp_path = None
        try:
            import wave
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
                with wave.open(tmp_path, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_bytes)

            config = aai.TranscriptionConfig(speaker_labels=True)
            transcriber = aai.Transcriber(config=config)

            import asyncio
            transcript = await asyncio.to_thread(transcriber.transcribe, tmp_path)

            if transcript.status == "error":
                logger.warning("[Speaker] Diarization failed: %s", transcript.error)
                return None

            if not transcript.utterances:
                logger.info("[Speaker] No utterances returned")
                return SpeakerAnalysis(speaker_count=1, second_speaker_pct=0.0, utterances_by_speaker={})

            # Compute duration per speaker
            speaker_durations: dict[str, float] = {}
            for utt in transcript.utterances:
                speaker = utt.speaker or "unknown"
                duration_ms = (utt.end or 0) - (utt.start or 0)
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration_ms

            total_duration = sum(speaker_durations.values())
            if total_duration == 0:
                return SpeakerAnalysis(speaker_count=1, second_speaker_pct=0.0, utterances_by_speaker={})

            primary = max(speaker_durations, key=speaker_durations.get)
            primary_duration = speaker_durations[primary]
            second_speaker_duration = total_duration - primary_duration
            second_speaker_pct = (second_speaker_duration / total_duration) * 100

            utterances_seconds = {s: d / 1000.0 for s, d in speaker_durations.items()}

            return SpeakerAnalysis(
                speaker_count=len(speaker_durations),
                second_speaker_pct=round(second_speaker_pct, 1),
                utterances_by_speaker=utterances_seconds,
            )

        except Exception as e:
            logger.warning("[Speaker] Diarization error: %s", e)
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
```

### Step 3: Add audio buffering to `RealtimeSTT`

**File:** `services/stt/stt_realtime.py`

Buffer raw audio chunks alongside existing streaming:

```python
class RealtimeSTT:
    def __init__(self):
        # ... existing init ...
        self.audio_buffer: bytearray = bytearray()  # NEW

    def stream(self, audio_base64: str):
        if self.client:
            audio_bytes = base64.b64decode(audio_base64)
            self.audio_buffer.extend(audio_bytes)  # NEW
            self.client.stream(audio_bytes)

    def get_buffered_audio(self) -> bytes:          # NEW
        """Return all buffered audio."""
        return bytes(self.audio_buffer)

    def close(self):
        # existing close logic...
        # NOTE: don't clear audio_buffer here — needed for diarization after close
```

### Step 4: Add `audioBuffer` field to `Session`

**File:** `services/interview/session_manager.py`

```python
@dataclass
class Session:
    # ... existing fields ...
    audioBuffer: bytearray = field(default_factory=bytearray)
```

### Step 5: Wire audio buffer from STT WebSocket to Session

**File:** `routers/transcription.py`

Accept `session_id` query param so audio can be linked to the interview session:

```python
@router.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket, session_id: str = None):
    # ... existing connection logic ...

    # At cleanup (finally block):
    finally:
        if stt:
            if session_id:
                from routers.interview import session_manager
                session = session_manager._sessions.get(session_id)
                if session:
                    session.audioBuffer.extend(stt.get_buffered_audio())
            stt.close()
```

**Frontend change needed**: Pass `session_id` when connecting to STT WebSocket:
```
Current:  ws://host/api/stt/realtime
Updated:  ws://host/api/stt/realtime?session_id=<sessionId>
```

### Step 6: Update `CheatDetector` with speaker signal

**File:** `services/interview/cheat_detector.py`

Add `speaker_count` and `second_speaker_pct` to `classify()` and `_determine_level()`:

```python
def classify(
    self,
    tab_count: int = 0,
    no_face_pct: float | None = None,
    multiple_face_pct: float | None = None,
    gaze_away_pct: float | None = None,
    speaker_count: int | None = None,           # NEW
    second_speaker_pct: float | None = None,    # NEW
) -> CheatClassification:
    evidence = CheatEvidence(
        tabSwitches=tab_count,
        noFacePct=no_face_pct,
        multipleFacePct=multiple_face_pct,
        gazeAwayPct=gaze_away_pct,
        speakerCount=speaker_count,
        secondSpeakerPct=second_speaker_pct,
    )
    level = self._determine_level(
        tab_count, no_face_pct, multiple_face_pct, gaze_away_pct,
        speaker_count, second_speaker_pct,
    )
    return CheatClassification(level=level, evidence=evidence)
```

In `_determine_level()`, add:

```python
if speaker_count is not None and speaker_count >= 2 and second_speaker_pct is not None:
    if second_speaker_pct > 15:
        critical_count += 1
    elif second_speaker_pct > 5:
        flagged_count += 1
```

### Step 7: Run diarization in `end_session()`

**File:** `services/interview/session_manager.py`

In `end_session()`, after video scoring but before cheat classification:

```python
# Speaker diarization
speaker_count = None
second_speaker_pct = None

if session.audioBuffer:
    try:
        from services.scoring.speaker_analyzer import SpeakerAnalyzer
        speaker_analyzer = SpeakerAnalyzer()
        speaker_result = await speaker_analyzer.analyze(bytes(session.audioBuffer))
        if speaker_result:
            speaker_count = speaker_result.speaker_count
            second_speaker_pct = speaker_result.second_speaker_pct
            if speaker_result.speaker_count > 1:
                logger.warning(
                    "Session %s: %d speakers detected, second speaker %.1f%%",
                    session_id, speaker_result.speaker_count, speaker_result.second_speaker_pct,
                )
    except Exception as e:
        logger.warning("Speaker diarization failed for session %s: %s", session_id, e)
```

Pass to `classify()`:

```python
cheat = cheat_detector.classify(
    tab_count=session.tabSwitches,
    no_face_pct=cheat_metrics.get("noFacePct"),
    multiple_face_pct=cheat_metrics.get("multipleFacePct"),
    gaze_away_pct=cheat_metrics.get("gazeAwayPct"),
    speaker_count=speaker_count,
    second_speaker_pct=second_speaker_pct,
)
```

### Step 8: Clear audio buffer after use

At end of `end_session()`, before `_remove_session()`:

```python
session.audioBuffer = bytearray()  # free memory
```

---

## Tests

### `tests/unit/test_speaker_analyzer.py` (NEW FILE)

```python
import pytest
from unittest.mock import MagicMock, patch
from services.scoring.speaker_analyzer import SpeakerAnalyzer, SpeakerAnalysis


class TestSpeakerAnalysis:
    def test_single_speaker(self):
        result = SpeakerAnalysis(speaker_count=1, second_speaker_pct=0.0, utterances_by_speaker={"A": 120.0})
        assert result.speaker_count == 1
        assert result.second_speaker_pct == 0.0

    def test_two_speakers(self):
        result = SpeakerAnalysis(speaker_count=2, second_speaker_pct=25.0, utterances_by_speaker={"A": 90.0, "B": 30.0})
        assert result.speaker_count == 2
        assert result.second_speaker_pct == 25.0

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_audio(self):
        analyzer = SpeakerAnalyzer()
        result = await analyzer.analyze(b"")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_short_audio(self):
        analyzer = SpeakerAnalyzer()
        result = await analyzer.analyze(b"\x00" * 100)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_api_failure(self):
        analyzer = SpeakerAnalyzer()
        with patch("services.scoring.speaker_analyzer.aai") as mock_aai:
            mock_transcriber = MagicMock()
            mock_aai.Transcriber.return_value = mock_transcriber
            mock_transcriber.transcribe.side_effect = Exception("API down")
            result = await analyzer.analyze(b"\x00" * 64000)
            assert result is None
```

### `tests/unit/test_cheat_detector.py` (ADD to existing)

```python
class TestCheatDetectorSpeakerDiarization:
    def test_clean_single_speaker(self):
        result = CheatDetector().classify(tab_count=0, speaker_count=1, second_speaker_pct=0.0)
        assert result.level == "Clean"

    def test_clean_below_5_pct(self):
        """Background noise — under 5%."""
        result = CheatDetector().classify(tab_count=0, speaker_count=2, second_speaker_pct=3.0)
        assert result.level == "Clean"

    def test_flagged_5_to_15_pct(self):
        result = CheatDetector().classify(tab_count=0, speaker_count=2, second_speaker_pct=10.0)
        assert result.level == "Flagged"

    def test_critical_above_15_pct(self):
        result = CheatDetector().classify(tab_count=0, speaker_count=2, second_speaker_pct=20.0)
        assert result.level == "Critical"

    def test_speaker_plus_tabs_escalates(self):
        """Speaker flagged + tabs flagged = Critical."""
        result = CheatDetector().classify(tab_count=4, speaker_count=2, second_speaker_pct=8.0)
        assert result.level == "Critical"

    def test_none_speaker_no_effect(self):
        """Diarization unavailable — no impact."""
        result = CheatDetector().classify(tab_count=0, speaker_count=None, second_speaker_pct=None)
        assert result.level == "Clean"

    def test_evidence_includes_speaker_fields(self):
        result = CheatDetector().classify(tab_count=0, speaker_count=2, second_speaker_pct=12.5)
        assert result.evidence.speakerCount == 2
        assert result.evidence.secondSpeakerPct == 12.5

    def test_three_speakers_critical(self):
        result = CheatDetector().classify(tab_count=0, speaker_count=3, second_speaker_pct=30.0)
        assert result.level == "Critical"
```

### `tests/unit/test_session_manager.py` (ADD to existing)

```python
class TestSpeakerDiarizationInEndSession:
    @pytest.mark.asyncio
    async def test_end_session_no_audio_buffer(self):
        """No audio buffer → no speaker signal → Clean."""
        mgr = SessionManager()
        session = mgr.create_session(mock_id="m1", candidate_id="c1", cv_url="https://cv.example.com")
        result = await mgr.end_session(session.id)
        assert result.cheat.evidence.speakerCount is None
        assert result.cheat.evidence.secondSpeakerPct is None
        assert result.cheat.level == "Clean"
```

### `tests/unit/test_stt.py` (ADD to existing)

```python
class TestRealtimeSTTAudioBuffer:
    def test_stream_buffers_audio(self):
        stt = RealtimeSTT.__new__(RealtimeSTT)
        stt.client = MagicMock()
        stt.audio_buffer = bytearray()

        import base64
        chunk1 = base64.b64encode(b"\x00\x01\x02\x03").decode()
        chunk2 = base64.b64encode(b"\x04\x05\x06\x07").decode()

        stt.stream(chunk1)
        stt.stream(chunk2)

        assert len(stt.audio_buffer) == 8
        assert stt.audio_buffer == bytearray(b"\x00\x01\x02\x03\x04\x05\x06\x07")

    def test_get_buffered_audio_returns_bytes(self):
        stt = RealtimeSTT.__new__(RealtimeSTT)
        stt.audio_buffer = bytearray(b"\x00\x01\x02")
        result = stt.get_buffered_audio()
        assert isinstance(result, bytes)
        assert len(result) == 3
```

---

## File Summary

| File | Action | What |
|---|---|---|
| `models/interview.py` | MODIFY | Add `speakerCount`, `secondSpeakerPct` to `CheatEvidence` |
| `services/scoring/speaker_analyzer.py` | **NEW** | AssemblyAI batch diarization service |
| `services/stt/stt_realtime.py` | MODIFY | Add `audio_buffer`, `get_buffered_audio()` |
| `services/interview/session_manager.py` | MODIFY | Add `audioBuffer` to Session, run diarization in `end_session()` |
| `services/interview/cheat_detector.py` | MODIFY | Add speaker signals to `classify()` and `_determine_level()` |
| `routers/transcription.py` | MODIFY | Accept `session_id` param, copy buffer to session on close |
| `tests/unit/test_speaker_analyzer.py` | **NEW** | Tests for SpeakerAnalyzer |
| `tests/unit/test_cheat_detector.py` | MODIFY | Add speaker diarization tests |
| `tests/unit/test_session_manager.py` | MODIFY | Add audio buffer end_session test |
| `tests/unit/test_stt.py` | MODIFY | Add audio buffer tests |

## Implementation Order

1. `CheatEvidence` model (no dependencies)
2. `SpeakerAnalyzer` service (no dependencies)
3. Audio buffering in `RealtimeSTT` (no dependencies)
4. `audioBuffer` field on Session (no dependencies)
5. Wire STT WebSocket to session (depends on 3, 4)
6. Update `CheatDetector` (depends on 1)
7. Wire diarization in `end_session()` (depends on 2, 4, 6)
8. Tests (after all implementation)

## Frontend Change (minimal)

Pass `session_id` when connecting to STT WebSocket:
```
Current:  ws://host/api/stt/realtime
Updated:  ws://host/api/stt/realtime?session_id=<sessionId>
```

## Memory & Cost

| Interview | Buffer Size | Notes |
|---|---|---|
| 15 min | ~29 MB | PCM 16-bit mono @ 16kHz = 32KB/s |
| 30 min | ~58 MB | Cleared at session end |
| 60 min | ~115 MB | Acceptable for server-side |

One AssemblyAI batch API call per interview session (billed per audio minute).
