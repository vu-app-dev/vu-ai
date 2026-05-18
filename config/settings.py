from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    
    SampleRate: int = 16000
    Language: str = "en"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    def validate(self):
        if not self.ASSEMBLYAI_API_KEY:
            raise ValueError("ASSEMBLYAI_API_KEY is not set in environment variables.")

settings = Settings()
settings.validate()