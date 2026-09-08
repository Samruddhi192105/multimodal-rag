"""
ingest.py
Pipeline that takes PDFs sitting in sample_docs/, pulls out both the text
and the embedded images, chunks the text, embeds everything with CLIP,
and writes it all into a persistent Chroma collection.

Run directly:  python ingest.py
"""

import io
import uuid
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from PIL import Image

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    IMAGE_STORE_DIR,
)
from embeddings import ClipEmbedder


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window character chunker."""
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def extract_from_pdf(pdf_path: Path):
    """Returns (text_chunks, image_records) for one PDF."""
    doc = fitz.open(pdf_path)
    text_chunks = []
    image_records = []

    for page_num, page in enumerate(doc, start=1):
        # --- text ---
        page_text = page.get_text()
        for chunk in chunk_text(page_text):
            text_chunks.append({"text": chunk, "page": page_num})

        # --- images ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                continue  # skip unreadable/corrupt embedded images

            # discard tiny decorative images (icons, bullets, etc.)
            if pil_image.width < 80 or pil_image.height < 80:
                continue

            filename = f"{pdf_path.stem}_p{page_num}_{img_index}.{ext}"
            out_path = IMAGE_STORE_DIR / filename
            pil_image.save(out_path)

            image_records.append(
                {"path": str(out_path), "page": page_num, "pil_image": pil_image}
            )

    doc.close()
    return text_chunks, image_records


def build_index(source_dir: Path = DATA_DIR, reset: bool = True):
    embedder = ClipEmbedder()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        # Drop any existing collection first so re-running ingest.py
        # replaces the index instead of duplicating every chunk on top
        # of what's already there.
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet — nothing to clear

    collection = client.get_or_create_collection(COLLECTION_NAME)

    pdf_files = list(Path(source_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {source_dir}. Add some and re-run.")
        return

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name} ...")
        text_chunks, image_records = extract_from_pdf(pdf_path)

        # --- index text chunks ---
        if text_chunks:
            texts = [c["text"] for c in text_chunks]
            vectors = embedder.embed_text(texts)
            ids = [str(uuid.uuid4()) for _ in texts]
            metadatas = [
                {
                    "modality": "text",
                    "source": pdf_path.name,
                    "page": c["page"],
                    "content": c["text"],
                }
                for c in text_chunks
            ]
            collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
            print(f"  indexed {len(texts)} text chunks")

        # --- index images ---
        if image_records:
            pil_images = [r["pil_image"] for r in image_records]
            vectors = embedder.embed_images(pil_images)
            ids = [str(uuid.uuid4()) for _ in image_records]
            metadatas = [
                {
                    "modality": "image",
                    "source": pdf_path.name,
                    "page": r["page"],
                    "content": r["path"],  # path to the saved image file
                }
                for r in image_records
            ]
            collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
            print(f"  indexed {len(image_records)} images")

    print("Done. Collection size:", collection.count())


if __name__ == "__main__":
    build_index()
