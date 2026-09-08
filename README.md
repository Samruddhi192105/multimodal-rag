# 📚 Multimodal RAG

A **Multimodal Retrieval-Augmented Generation (RAG)** application that
retrieves and reasons over **both text and images from PDF documents**.

The system uses **CLIP** to place text and images into a shared
embedding space, **ChromaDB** for vector retrieval, and **Google
Gemini** to generate grounded answers using the retrieved text and
images.

## 🚀 Live Demo

**Try the deployed application:**\
https://multimodal-rag-web2.streamlit.app/

## 📸 Application

![Multimodal RAG application](assets/app-screenshot.png)

## 🧠 How It Works

``` text
                    PDF
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     Page Text               Images
          │                     │
          ▼                     ▼
     Text Chunking        Image Extraction
     500 chars             PyMuPDF
     80 overlap                 │
          │                     │
          └──────────┬──────────┘
                     ▼
              CLIP Embeddings
          Text + Images → shared space
                     │
                     ▼
                 ChromaDB
                     │
               User Question
                     │
                     ▼
             CLIP Text Embedding
                     │
                     ▼
                Top-K Retrieval
             ┌───────┴───────┐
             ▼               ▼
          Text Chunks      Images
             │               │
             └───────┬───────┘
                     ▼
              Google Gemini
                     │
                     ▼
               Grounded Answer
```

### Page-specific retrieval

For questions such as:

> "Tell me about page number 24"

the application detects the requested page and retrieves the indexed
content for that specific page instead of relying on semantic top-k
search.

For normal questions, the system performs **top-k similarity retrieval**
across the selected PDF.

## ✨ Key Features

-   📄 Upload one or more PDF documents
-   📝 Extract text from PDF pages using PyMuPDF
-   🖼️ Extract embedded PDF images
-   ✂️ Split text into overlapping chunks
-   🔎 Generate shared text/image embeddings using CLIP
-   🗄️ Store embeddings and metadata in ChromaDB
-   🎯 Retrieve the top-k relevant text chunks and images
-   📑 Support page-specific retrieval such as "page 24"
-   🤖 Generate grounded answers with Google Gemini
-   💬 Interactive Streamlit chat interface
-   🌐 Deployed on Streamlit Community Cloud

## 🛠️ Tech Stack

  Component               Technology
  ----------------------- ----------------------------------------------
  UI                      Streamlit
  PDF processing          PyMuPDF
  Text/image embeddings   OpenAI CLIP (`openai/clip-vit-base-patch32`)
  Vector database         ChromaDB
  Generation              Google Gemini
  Image processing        Pillow
  Language                Python

## 📁 Project Structure

``` text
multimodal-rag/
├── app.py                 # Streamlit user interface
├── config.py              # Configuration and model settings
├── embeddings.py          # CLIP text/image embedding wrapper
├── ingest.py              # PDF extraction, chunking and indexing
├── rag_pipeline.py        # Retrieval and Gemini generation
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Ignored files and secrets
├── assets/
│   └── app-screenshot.png # Application screenshot
└── sample_docs/
    └── .gitkeep
```

## ⚙️ Local Setup

### 1. Clone the repository

``` bash
git clone https://github.com/Samruddhi192105/multimodal-rag.git
cd multimodal-rag
```

### 2. Create a virtual environment

``` bash
python -m venv venv
```

Activate it:

**Windows**

``` bash
venv\Scripts\activate
```

**macOS / Linux**

``` bash
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Copy the example environment file:

**Windows**

``` bash
copy .env.example .env
```

**macOS / Linux**

``` bash
cp .env.example .env
```

Add your Gemini API key to `.env`:

``` env
GEMINI_API_KEY=your_api_key_here
```

**Never commit `.env` or your actual API key to GitHub.**

### 5. Add PDFs

Place your PDF files inside:

``` text
sample_docs/
```

PDFs containing diagrams, charts, or other images are especially useful
for demonstrating the multimodal retrieval capability.

### 6. Build the index

``` bash
python ingest.py
```

The ingestion pipeline:

1.  Opens each PDF.
2.  Extracts page text.
3.  Creates overlapping text chunks.
4.  Extracts embedded images.
5.  Generates CLIP embeddings for text and images.
6.  Stores embeddings and metadata in ChromaDB.

### 7. Run the application

``` bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

You can also upload PDFs directly from the application's sidebar and
click **Build / refresh index**.

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

### Deployment configuration

``` text
Repository:  Samruddhi192105/multimodal-rag
Branch:      main
Main file:   app.py
```

Add the Gemini API key through the deployment platform's **Secrets**
settings rather than committing it to the repository.

### Live application

https://multimodal-rag-web2.streamlit.app/

> The deployed environment does not rely on the Chroma index generated
> on your local machine. Upload the required PDF through the application
> and build/refresh the index in the deployed environment.

## 🔐 Security

The Gemini API key is kept outside the repository.

The following are excluded using `.gitignore`:

``` text
.env
venv/
chroma_db/
extracted_images/
__pycache__/
```

Only `.env.example` is committed as a configuration template.

## 🔬 Retrieval Strategy

### Normal questions

``` text
Question
   ↓
CLIP text embedding
   ↓
ChromaDB similarity search
   ↓
Top-k text/image results
   ↓
Gemini
   ↓
Grounded answer
```

### Page-specific questions

``` text
"Tell me about page 24"
             ↓
     Detect page number
             ↓
       Page = 24
             ↓
  Exact page + source lookup
             ↓
      Retrieved context
             ↓
           Gemini
             ↓
        Grounded answer
```

This combination allows the system to support both **semantic
retrieval** and **direct page-based retrieval**.

## 🔮 Future Improvements

-   Add PDF table extraction as another information modality
-   Add a re-ranking stage for improved retrieval precision
-   Support more advanced multimodal embedding models
-   Add conversation-aware retrieval
-   Add evaluation using Precision@K, Recall@K, MRR, answer accuracy,
    and hallucination rate
-   Use a hosted vector database for persistent cloud indexing
-   Add authentication and multi-user document collections

## 👩‍💻 Author

**Samruddhi**

GitHub:\
https://github.com/Samruddhi192105
