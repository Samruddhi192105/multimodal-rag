# Multimodal RAG

A Retrieval-Augmented Generation system that indexes **both text and images**
from PDFs into one shared vector space (via CLIP), retrieves the most
relevant mix of text chunks and images for a question, and generates a
grounded answer using Google Gemini (a multimodal LLM that can read the
retrieved images directly).

---

## 1. Architecture

```
PDF files
   │
   ├─► PyMuPDF extracts page text ──► chunk (500 chars, 80 overlap)
   └─► PyMuPDF extracts embedded images (filters out tiny icons)
                        │
                        ▼
              CLIP (openai/clip-vit-base-patch32)
        embeds text chunks AND images into ONE shared space
                        │
                        ▼
                  Chroma (persistent, local)
                        │
      user query ──► CLIP text-embed ──► top-k nearest vectors
                        │  (mix of text chunks + images)
                        ▼
        Gemini 2.0 Flash receives the text context AND the
        actual retrieved images, then answers the question
```

Why CLIP for embeddings: it puts text and images in the *same* vector
space, so a text query can retrieve a relevant image and vice versa —
that's what makes this "multimodal" rather than just RAG with an image
captioning step bolted on.

## 2. Project structure

```
multimodal-rag/
├── app.py              # Streamlit chat UI
├── ingest.py            # PDF → text/image extraction → embeddings → Chroma
├── rag_pipeline.py       # retrieval + Gemini generation
├── embeddings.py          # CLIP wrapper (shared text/image embedder)
├── config.py               # paths, model names, chunking/retrieval params
├── requirements.txt
├── .env.example
├── .gitignore
└── sample_docs/            # drop your PDFs here
```

---

## 3. Prerequisites

- Python 3.10+
- VS Code with the **Python extension** (ms-python.python)
- A GitHub account
- A free Gemini API key: https://aistudio.google.com/app/apikey

---

## 4. Local setup in VS Code (step by step)

### Step 1 — Open the project
Open the `multimodal-rag` folder in VS Code: `File → Open Folder...`

### Step 2 — Create a virtual environment
Open the VS Code integrated terminal (`` Ctrl+` ``) and run:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

VS Code will usually pop up a prompt asking "Select this environment for
the workspace?" — click **Yes**. If it doesn't, open the Command Palette
(`Ctrl+Shift+P`) → `Python: Select Interpreter` → pick the one inside
`venv`.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add your API key
Copy the template and fill it in:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and paste your key:

```
GEMINI_API_KEY=AIza...your_real_key...
```

### Step 5 — Add documents
Drop one or more PDFs into `sample_docs/` (PDFs with diagrams/photos in
them show off the multimodal retrieval best).

### Step 6 — Build the index

```bash
python ingest.py
```

You should see output like:

```
Processing my_paper.pdf ...
  indexed 42 text chunks
  indexed 6 images
Done. Collection size: 48
```

### Step 7 — Run the app

```bash
streamlit run app.py
```

This opens `http://localhost:8501` in your browser. You can also use the
sidebar's "Upload PDF(s)" + "Build / refresh index" instead of steps 5–6.

---

## 5. Pushing to GitHub (step by step)

### Step 1 — Initialize git
In the VS Code terminal, inside `multimodal-rag/`:

```bash
git init
git add .
git commit -m "Initial commit: multimodal RAG pipeline"
```

`.env`, `venv/`, `chroma_db/`, and extracted images are already excluded
via `.gitignore` — never commit your API key.

### Step 2 — Create the GitHub repo
Go to https://github.com/new, name it (e.g. `multimodal-rag`), leave it
empty (no README/license — you already have files), and click **Create
repository**.

### Step 3 — Connect and push

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/multimodal-rag.git
git push -u origin main
```

(VS Code's built-in **Source Control** panel, the icon on the left
sidebar, does the same thing with buttons if you prefer a GUI: Initialize
Repository → stage all → commit → Publish Branch.)

### Step 4 — Verify
Refresh your GitHub repo page — all files except the ignored ones should
be there.

---

## 6. Deployment (Streamlit Community Cloud — free)

### Step 1 — Go to Streamlit Cloud
Visit https://share.streamlit.io and sign in with GitHub.

### Step 2 — New app
Click **New app** → select your `multimodal-rag` repository → branch
`main` → main file path `app.py`.

### Step 3 — Add your secret
Before deploying, click **Advanced settings** → **Secrets**, and add:

```toml
GEMINI_API_KEY = "AIza...your_real_key..."
```

This is the cloud equivalent of your local `.env` file — Streamlit Cloud
injects it as an environment variable at runtime, so `config.py` picks it
up the same way via `os.getenv`.

### Step 4 — Deploy
Click **Deploy**. First build takes a few minutes (installing torch and
transformers). Once live, you'll get a public URL like
`https://your-app-name.streamlit.app`.

### Step 5 — Re-indexing on the deployed app
Streamlit Cloud's filesystem is ephemeral, so the index built on your
laptop doesn't travel with the repo. Upload your PDFs through the app's
sidebar and click "Build / refresh index" once after each deploy/restart.

> **Note on free-tier limits:** the CLIP model (~600MB) and torch are
> heavy for Streamlit Cloud's free 1GB RAM tier. If you hit memory
> errors, switch `CLIP_MODEL_NAME` in `config.py` to a smaller model such
> as `"openai/clip-vit-base-patch16"`, or deploy instead to
> [Hugging Face Spaces](https://huggingface.co/spaces) (free tier, more
> RAM, built specifically for ML apps — choose the "Streamlit" SDK when
> creating the Space, then push this same repo to it).

---

## 7. Alternative deployment: Hugging Face Spaces

```bash
# after creating a new Space (SDK: Streamlit) on huggingface.co
git remote add hf https://huggingface.co/spaces/<your-username>/multimodal-rag
git push hf main
```

Add `GEMINI_API_KEY` under the Space's **Settings → Repository secrets**.

---

## 8. Extending this project

- Swap Gemini for a local model (LLaVA via Ollama) for a fully offline
  pipeline.
- Add PDF tables as a third modality (extract with `camelot` or
  `pdfplumber`, embed as text).
- Add re-ranking: retrieve top-20 with CLIP, re-rank top-5 with a
  cross-encoder for higher precision.
- Swap Chroma for a hosted vector DB (Pinecone/Weaviate) if you need
  persistence across ephemeral cloud filesystems without re-indexing.
