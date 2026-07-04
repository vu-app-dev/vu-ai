from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import transcription_router, interview_router, cv_router

print(f"[VU-AI] LLM Provider: {settings.LLM_PROVIDER} | Model: {settings.LLM_MODEL or '(default)'}")

app = FastAPI(
    title="VU - AI Interview Assistant",
    description="AI-powered interview service with STT, LLM evaluation, and video analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=settings.CORS_ORIGINS
    or r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Session-Token", "X-API-Key"],
)

app.include_router(transcription_router)
app.include_router(interview_router)
app.include_router(cv_router)


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
        reload=False,
    )