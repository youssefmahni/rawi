import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
STT_ENGINE = os.getenv("STT_ENGINE", "elevenlabs")  # elevenlabs (cloud Scribe) | whisper (local faster-whisper)
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2")
STT_MODEL = os.getenv("STT_MODEL", "") or os.getenv("WHISPER_MODEL", "") or "medium"  # local Whisper only
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")  # cpu | cuda | auto
STT_COMPUTE = os.getenv("STT_COMPUTE", "")  # empty → int8 on CPU, float16 on CUDA
STT_CPU_THREADS = int(os.getenv("STT_CPU_THREADS", "") or min(8, os.cpu_count() or 4))
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
VOICES_FILE = os.getenv("VOICES_FILE", str(BASE_DIR / "voices" / "voices.json"))
PORT = int(os.getenv("PORT", "8765"))

HOST = os.getenv("HOST") or "127.0.0.1"
ALLOWED_HOSTS = [h for h in os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if h]
SSL_CERTFILE = os.getenv("SSL_CERTFILE", "")
SSL_KEYFILE = os.getenv("SSL_KEYFILE", "")


def allowed_origins() -> set[str]:
    origins = set()
    for host in ALLOWED_HOSTS:
        origins.add(f"http://{host}:{PORT}")
        if SSL_CERTFILE and SSL_KEYFILE:
            origins.add(f"https://{host}:{PORT}")
    return origins


def uvicorn_ssl_kwargs() -> dict[str, str]:
    if SSL_CERTFILE and SSL_KEYFILE:
        return {"ssl_certfile": SSL_CERTFILE, "ssl_keyfile": SSL_KEYFILE}
    return {}
