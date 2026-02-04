from fastapi import FastAPI
from fastapi.responses import FileResponse

from config import settings
from routers import transcription_router

app = FastAPI(
    title="VU - AI Interview Assistant",
    description="Speech-to-Text service for AI-powered interviews",
    version="1.0.0"
)

app.include_router(transcription_router)


@app.get("/")
async def serve_frontend():
    """Serve the test page"""
    return FileResponse("templates/index.html")


@app.get("/health")
async def health_check():
    """Check if server is running"""
    return {"status": "ok", "service": "stt"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
    