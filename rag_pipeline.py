"""
rag_pipeline.py

Retrieval + generation.

- Normal questions -> top-k multimodal retrieval
- Questions like "Tell me about page 48" -> exact page retrieval
- Retrieved text + images are sent to Gemini
"""

from pathlib import Path
import re

import chromadb
import google.generativeai as genai
from PIL import Image

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    TOP_K,
)
from embeddings import ClipEmbedder


if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class MultimodalRAG:

    def __init__(self):
        # Load CLIP
        self.embedder = ClipEmbedder()

        # Connect to Chroma
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        self.collection = client.get_or_create_collection(
            COLLECTION_NAME
        )

        # Load Gemini
        self.model = (
            genai.GenerativeModel(GEMINI_MODEL_NAME)
            if GEMINI_API_KEY
            else None
        )

    # ---------------------------------------------------------
    # NORMAL TOP-K RETRIEVAL
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = TOP_K,
        source: str | None = None,
    ) -> list[dict]:

        # Convert user question into CLIP vector
        query_vector = self.embedder.embed_single_text(query)

        # Prepare Chroma query
        query_args = {
            "query_embeddings": [query_vector],
            "n_results": k,
        }

        # If a PDF is selected, restrict retrieval to that PDF
        if source and source != "(none indexed)":
            query_args["where"] = {
                "source": source
            }

        results = self.collection.query(**query_args)

        retrieved = []

        for metadata, distance in zip(
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                {
                    **metadata,
                    "score": 1 - distance,
                }
            )

        return retrieved

    # ---------------------------------------------------------
    # EXACT PAGE RETRIEVAL
    # ---------------------------------------------------------

    def retrieve_page(
        self,
        source: str,
        page: int,
    ) -> list[dict]:

        results = self.collection.get(
            where={
                "$and": [
                    {"source": source},
                    {"page": page},
                ]
            }
        )

        retrieved = []

        for metadata in results["metadatas"]:
            retrieved.append(
                {
                    **metadata,
                    "score": 1.0,
                }
            )

        return retrieved

    # ---------------------------------------------------------
    # BUILD GEMINI PROMPT
    # ---------------------------------------------------------

    def _build_prompt_parts(
        self,
        query: str,
        retrieved: list[dict],
    ) -> list:

        context_lines = [
            "Context retrieved from the document collection:\n"
        ]

        parts = []

        # Add text/image information to textual context
        for item in retrieved:

            if item["modality"] == "text":

                context_lines.append(
                    f"- (source: {item['source']}, "
                    f"page {item['page']}) "
                    f"{item['content']}"
                )

            else:

                context_lines.append(
                    f"- (source: {item['source']}, "
                    f"page {item['page']}) "
                    "[see attached image]"
                )

        parts.append("\n".join(context_lines))

        # Attach actual images to Gemini
        for item in retrieved:

            if item["modality"] == "image":

                img_path = Path(item["content"])

                if img_path.exists():

                    parts.append(
                        Image.open(img_path)
                    )

        # Instructions for Gemini
        instructions = (
            "\n\nUsing ONLY the context and images above, "
            "answer the question. "
            "If the answer isn't in the context, say so.\n\n"
            f"Question: {query}"
        )

        parts.append(instructions)

        return parts

    # ---------------------------------------------------------
    # MAIN ANSWER FUNCTION
    # ---------------------------------------------------------

    def answer(
        self,
        query: str,
        k: int = TOP_K,
        source: str | None = None,
    ) -> dict:

        # Check Gemini API key
        if not self.model:

            return {
                "answer": (
                    "GEMINI_API_KEY is not set. "
                    "Add it to your .env file."
                ),
                "retrieved": [],
            }

        # -----------------------------------------------------
        # FIRST: CHECK IF USER ASKED FOR A SPECIFIC PAGE
        # -----------------------------------------------------

        page_match = re.search(
            r"\bpage\s+(?:number\s+)?(\d+)\b",
            query,
            re.IGNORECASE,
        )

        if page_match and source:

            # Extract page number
            page_num = int(page_match.group(1))

            print(
                f"Page-specific query detected: "
                f"page {page_num}"
            )

            # Retrieve ONLY that page
            retrieved = self.retrieve_page(
                source,
                page_num,
            )

            # Nothing found on that page
            if not retrieved:

                return {
                    "answer": (
                        f"Nothing was indexed for page "
                        f"{page_num} of {source}. "
                        "Either the page has no extractable "
                        "text/images or the page is outside "
                        "the document range."
                    ),
                    "retrieved": [],
                }

        # -----------------------------------------------------
        # OTHERWISE: NORMAL TOP-K RETRIEVAL
        # -----------------------------------------------------

        else:

            retrieved = self.retrieve(
                query,
                k=k,
                source=source,
            )

        # -----------------------------------------------------
        # NO RESULTS
        # -----------------------------------------------------

        if not retrieved:

            return {
                "answer": (
                    "No indexed documents found. "
                    "Build the index first."
                ),
                "retrieved": [],
            }

        # -----------------------------------------------------
        # SEND RETRIEVED CONTENT TO GEMINI
        # -----------------------------------------------------

        parts = self._build_prompt_parts(
            query,
            retrieved,
        )

        response = self.model.generate_content(parts)

        return {
            "answer": response.text,
            "retrieved": retrieved,
        }