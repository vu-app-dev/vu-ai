import asyncio
import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException

from config import settings
from services.stt.stt_service import STT
from services.stt.stt_realtime import RealtimeSTT, ASSEMBLYAI_AVAILABLE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stt", tags=["Speech-to-Text"])


@router.post("/transcribe/url")
async def transcribe_url(url: str):
    """Transcribe audio from a URL."""
    try:
        stt = STT()
        result = stt.transcribe(url)
        return {
            "text": result.text,
            "words": result.words,
            "duration": result.duration,
        }
    except Exception as e:
        logger.error("[STT URL] Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        stt = STT()
        result = stt.transcribe(tmp_path)
        return {
            "text": result.text,
            "words": result.words,
            "duration": result.duration,
        }
    except Exception as e:
        logger.error("[STT File] Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@router.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket):
    """Real-time STT WebSocket using AssemblyAI Universal Streaming v3.

    Protocol:
    - Client connects
    - Client sends: {"audio": "<base64 PCM 16kHz mono>"}
    - Client sends: {"type": "end_stream"} to signal end of audio
    - Server sends: {"type": "session_begins", "session_id": "..."}
    - Server sends: {"type": "partial", "text": "..."}
    - Server sends: {"type": "final", "text": "..."}
    - Server sends: {"type": "error", "message": "..."} on errors
    """
    await websocket.accept()
    logger.info("[STT WS] Client connected")

    if not ASSEMBLYAI_AVAILABLE:
        await websocket.send_json({"type": "error", "message": "AssemblyAI streaming not available"})
        await websocket.close()
        return

    stt: Optional[RealtimeSTT] = None

    try:
        loop = asyncio.get_event_loop()
        stt = RealtimeSTT()
        transcript_queue = stt.connect(loop)
        logger.info("[STT WS] AssemblyAI session connected")

        async def send_transcripts():
            while True:
                try:
                    data = await asyncio.wait_for(transcript_queue.get(), timeout=0.1)
                    await websocket.send_json(data)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

        send_task = asyncio.create_task(send_transcripts())

        try:
            while True:
                data = await websocket.receive_json()
                if "audio" in data:
                    stt.stream(data["audio"])
                elif data.get("type") == "end_stream":
                    stt.close()
                    await asyncio.sleep(0.5)
                    break
        except WebSocketDisconnect:
            logger.info("[STT WS] Client disconnected")

    except Exception as e:
        logger.error("[STT WS] Error: %s", e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if stt:
            stt.close()
        logger.info("[STT WS] Cleanup complete")