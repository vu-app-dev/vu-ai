from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import settings
from routers import transcription_router, interview_router, cv_router

app = FastAPI(
    title="VU - AI Interview Assistant",
    description="AI-powered interview service with STT, LLM evaluation, and video analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Session-Token", "X-API-Key"],
)

app.include_router(transcription_router)
app.include_router(interview_router)
app.include_router(cv_router)


@app.get("/")
async def serve_interview_test():
    """Serve the interview test page"""
    return FileResponse("templates/interview_test.html")


@app.get("/health")
async def health_check():
    """Check if server is running"""
    return {"status": "ok", "service": "vu-ai"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )