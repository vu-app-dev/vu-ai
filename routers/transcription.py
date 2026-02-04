from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
import asyncio
import tempfile
import os

from services import STT, RealtimeSTT

router = APIRouter(prefix="/api/stt", tags=["Speech-to-Text"])

@router.post("/transcribe/url")
async def transcribe_url(url: str):
    """
    Transcribe audio from a URL.
    
    Example: POST /api/stt/transcribe/url?url=https://example.com/audio.mp3
    """
    try:
        stt = STT()
        result = stt.transcribe(url)
        return {
            "text": result.text,
            "words": result.words,
            "duration": result.duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file.
    
    Example: POST /api/stt/transcribe/file with file in form data
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        stt = STT()
        result = stt.transcribe(tmp_path)
        
        os.unlink(tmp_path)
        
        return {
            "text": result.text,
            "words": result.words,
            "duration": result.duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription.
    
    Connect: ws://localhost:8000/api/stt/realtime
    Send: {"audio": "<base64 encoded audio>"}
    Receive: {"type": "partial|final", "text": "..."}
    """

    await websocket.accept()
    print("[WebSocket] Client connected")
    
    realtime_stt = RealtimeSTT()
    
    try:
        loop = asyncio.get_event_loop()
        transcript_queue = realtime_stt.connect(loop)
        print("[WebSocket] Connected to AssemblyAI")
        
        async def send_transcripts():
            """Continuously check queue and send to browser"""
            while True:
                try:
                    data = await asyncio.wait_for(
                        transcript_queue.get(), 
                        timeout=0.1
                    )
                    await websocket.send_json(data)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
        
        send_task = asyncio.create_task(send_transcripts())
        
        while True:
            data = await websocket.receive_json()
            if "audio" in data:
                realtime_stt.stream(data["audio"])
    
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
    finally:
        realtime_stt.close()
        print("[WebSocket] Cleanup complete")
        