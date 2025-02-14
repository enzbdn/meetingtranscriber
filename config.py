import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ASR_MODEL = os.getenv('ASR_MODEL', "openai/whisper-base")
    DIARIZATION_MODEL = os.getenv('DIARIZATION_MODEL', "pyannote/speaker-diarization@2.1")
    TEMP_DIR = "temp"
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'mp4', 'avi', 'mov'}

    WHISPER_API_URL = os.getenv('WHISPER_API_URL', "http://localhost:11433/asr")
    OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', "http://localhost:11434/api/generate")
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', "llama3.1")
    MAX_SPEAKERS = 10
    MIN_SPEAKERS = 1
    CHUNK_LENGTH = 30

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
