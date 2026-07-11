from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    mockId: str = ""
    candidateId: str
    cvUrl: str = ""
    mockData: Optional[dict] = None
    mocks: Optional[list[dict]] = None
    skipIntro: bool = False
    candidateIntro: str = ""
    previousQuestions: list[str] = Field(default_factory=list)


ALL_TRANSCRIPT_DIMENSIONS = [
    "communication", "problemSolving", "technical",
    "clarityOfExplanation", "structuredThinking",
]


class Question(BaseModel):
    id: str
    text: str
    difficulty: Literal["EASY", "MEDIUM", "HARD"]
    order: int
    speechType: Literal["question", "follow_up"] = "question"
    activeDimensions: list[str] | None = None
    topicTag: Optional[str] = None


class CheatEvidence(BaseModel):
    tabSwitches: int = 0
    noFacePct: Optional[float] = None
    multipleFacePct: Optional[float] = None
    gazeAwayPct: Optional[float] = None
    speakerCount: Optional[int] = None
    secondSpeakerPct: Optional[float] = None


class CheatClassification(BaseModel):
    level: Literal["Clean", "Flagged", "Critical"] = "Clean"
    evidence: CheatEvidence = Field(default_factory=CheatEvidence)


class StartSessionResponse(BaseModel):
    sessionId: str
    sessionToken: str
    intro: str
    introAudio: Optional[str] = None
    firstQuestion: Question
    firstQuestionAudio: Optional[str] = None
    cvAnalysis: Optional[dict] = None
    mockIndex: int = 0
    totalMocks: int = 1


class WSMessage(BaseModel):
    type: str
    messageId: str = Field(default_factory=lambda: str(uuid4()))
    sessionId: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WSAnswerMessage(WSMessage):
    type: Literal["answer"] = "answer"
    questionId: str
    transcript: str
    durationSeconds: int
    startedAt: str
    endedAt: str


class WSVideoFrameMessage(WSMessage):
    type: Literal["video_frame"] = "video_frame"
    image: str
    frameNumber: int


class WSTabSwitchMessage(WSMessage):
    type: Literal["tab_switch"] = "tab_switch"
    totalCount: int


class WSEndSessionMessage(WSMessage):
    type: Literal["end_session"] = "end_session"


class WSIntroMessage(WSMessage):
    type: Literal["intro"] = "intro"
    text: str
    speechType: Literal["intro"] = "intro"
    audioBase64: Optional[str] = None


class WSQuestionMessage(WSMessage):
    type: Literal["question"] = "question"
    id: str
    text: str
    difficulty: Literal["EASY", "MEDIUM", "HARD"]
    order: int
    speechType: Literal["question", "follow_up"] = "question"
    topicTag: Optional[str] = None
    audioBase64: Optional[str] = None


class WSAcknowledgementMessage(WSMessage):
    type: Literal["acknowledgement"] = "acknowledgement"
    text: str
    speechType: Literal["feedback"] = "feedback"
    audioBase64: Optional[str] = None


class WSCheatWarningMessage(WSMessage):
    type: Literal["cheat_warning"] = "cheat_warning"
    level: Literal["Flagged", "Critical"]
    reason: str
    evidenceSignals: list[str] = Field(default_factory=list)


class WSAnalysisUpdateMessage(WSMessage):
    type: Literal["analysis_update"] = "analysis_update"
    eyeContactScore: Optional[float] = None


class WSMockTimeWarningMessage(WSMessage):
    type: Literal["mock_time_warning"] = "mock_time_warning"
    mockIndex: int
    graceSeconds: int = 30
    message: str


class WSMockTransitionMessage(WSMessage):
    type: Literal["mock_transition"] = "mock_transition"
    mockIndex: int
    totalMocks: int
    mockId: str
    mockType: str
    reason: Literal["completed", "time_expired"]
    intro: str
    introAudio: Optional[str] = None
    firstQuestion: Question
    firstQuestionAudio: Optional[str] = None


class WSSessionEndMessage(WSMessage):
    type: Literal["session_end"] = "session_end"
    reason: Literal["completed", "time_expired"]
    performance: Optional[dict] = None
    cheat: str = "Clean"
    cheatEvidence: Optional[CheatEvidence] = None
    mockIndex: Optional[int] = None
    totalMocks: Optional[int] = None
    closingText: Optional[str] = None
    closingAudioBase64: Optional[str] = None


class WSErrorMessage(WSMessage):
    type: Literal["error"] = "error"
    code: Literal[
        "RATE_LIMITED",
        "SESSION_EXPIRED",
        "INVALID_MESSAGE",
        "PROCESSING_ERROR",
    ]
    message: str
    retryable: bool = False
