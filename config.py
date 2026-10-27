import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge_base"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
FAISS_DIR = ROOT / "faiss_index"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.5-flash")
OPENROUTER_TEXT_MODEL = os.getenv("OPENROUTER_TEXT_MODEL", "openrouter/free")

YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.30
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVAL_K = 4
