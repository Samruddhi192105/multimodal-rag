"""
config.py
Central configuration for the Multimodal RAG project.
All paths, model names, and tunable parameters live here so nothing
is hard-coded inside the pipeline logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file if present

# ---------- API keys ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "sample_docs"          # drop PDFs here
IMAGE_STORE_DIR = BASE_DIR / "extracted_images"  # images pulled out of PDFs
CHROMA_DIR = BASE_DIR / "chroma_db"          # persistent vector store

DATA_DIR.mkdir(exist_ok=True)
IMAGE_STORE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ---------- Models ----------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"   # shared text+image embedding space
GEMINI_MODEL_NAME = "gemini-3.6-flash"             # multimodal generation model

# ---------- Chunking ----------
CHUNK_SIZE = 500       # characters per text chunk
CHUNK_OVERLAP = 80     # characters of overlap between consecutive chunks

# ---------- Retrieval ----------
TOP_K = 5               # how many chunks/images to retrieve per query
COLLECTION_NAME = "multimodal_rag"
