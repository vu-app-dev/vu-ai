from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")

    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:3000")
    BACKEND_API_KEY: str = os.getenv("BACKEND_API_KEY", "")

    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "")

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    SAMPLE_RATE: int = 16000
    LANGUAGE: str = "en"

    # TTS (edge-tts — Microsoft neural voices, no API key needed)
    TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-AriaNeural")
    TTS_RATE: str = os.getenv("TTS_RATE", "")
    TTS_VOLUME: str = os.getenv("TTS_VOLUME", "")
    TTS_PITCH: str = os.getenv("TTS_PITCH", "")
    TTS_PROXY: str = os.getenv("TTS_PROXY", "")

    SESSION_TIMEOUT_SECONDS: int = int(os.getenv("SESSION_TIMEOUT_SECONDS", "120"))
    LLM_MAX_RETRIES: int = 3
    LLM_RPM_LIMIT: int = 15
    BACKEND_RETRY_MAX_ATTEMPTS: int = 5
    BACKEND_RETRY_INTERVAL_SECONDS: int = 30
    BACKEND_RETRY_QUEUE_MAX_SIZE: int = 50
    CV_MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024
    CV_DOWNLOAD_TIMEOUT_SECONDS: int = 30

    def validate(self):
        missing = []
        if not self.ASSEMBLYAI_API_KEY:
            missing.append("ASSEMBLYAI_API_KEY")
        if self.LLM_PROVIDER.lower() == "gemini" and not self.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if self.LLM_PROVIDER.lower() == "groq" and not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not self.GEMINI_API_KEY and not self.GROQ_API_KEY:
            missing.append("GEMINI_API_KEY or GROQ_API_KEY")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


settings = Settings()
settings.validate()