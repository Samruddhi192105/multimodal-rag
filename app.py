"""
app.py

Streamlit front end for Multimodal RAG.

- Upload PDFs
- Build / refresh index
- Select source PDF
- Ask questions
- Normal questions use top-k retrieval
- "page N" questions use exact page retrieval
"""

from pathlib import Path

import streamlit as st

from config import DATA_DIR
from ingest import build_index
from rag_pipeline import MultimodalRAG


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Multimodal RAG")

st.caption(
    "Upload PDFs, then ask questions grounded "
    "in their text and images."
)


# ---------------------------------------------------------
# CACHE PIPELINE
# ---------------------------------------------------------

@st.cache_resource
def get_pipeline():
    return MultimodalRAG()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("1. Add documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        for f in uploaded_files:

            out_path = Path(DATA_DIR) / f.name

            with open(out_path, "wb") as fh:
                fh.write(f.getbuffer())

        st.success(
            f"Saved {len(uploaded_files)} file(s) "
            f"to {DATA_DIR}"
        )

    # -----------------------------------------------------
    # BUILD INDEX
    # -----------------------------------------------------

    if st.button(
        "Build / refresh index",
        use_container_width=True,
    ):

        with st.spinner(
            "Extracting text + images and embedding with CLIP..."
        ):

            build_index()

            # IMPORTANT:
            # Remove the old cached pipeline because
            # build_index() recreated the Chroma collection.
            get_pipeline.clear()

        st.success("Index ready.")

    st.divider()

    # -----------------------------------------------------
    # TOP-K
    # -----------------------------------------------------

    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=10,
        value=5,
    )

    # -----------------------------------------------------
    # SOURCE PDF
    # -----------------------------------------------------

    indexed_pdfs = [
        f.name
        for f in Path(DATA_DIR).glob("*.pdf")
    ]

    selected_source = st.selectbox(
        "Source PDF",
        indexed_pdfs or ["(none indexed)"],
        help=(
            "Normal questions are searched within "
            "this selected PDF. Questions such as "
            "'Tell me about page 48' retrieve that "
            "specific page."
        ),
    )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

if "history" not in st.session_state:

    st.session_state.history = []


for msg in st.session_state.history:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        for img_path in msg.get("images", []):

            st.image(
                img_path,
                width=250,
            )


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

query = st.chat_input(
    "Ask a question about your documents..."
)


if query:

    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    st.session_state.history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving context and generating answer..."
        ):

            pipeline = get_pipeline()

            result = pipeline.answer(
                query=query,
                k=top_k,
                source=selected_source,
            )

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        st.markdown(
            result["answer"]
        )

        # -------------------------------------------------
        # DISPLAY RETRIEVED IMAGES
        # -------------------------------------------------

        image_paths = [
            item["content"]
            for item in result["retrieved"]
            if item["modality"] == "image"
        ]

        for path in image_paths:

            st.image(
                path,
                width=250,
            )

        # -------------------------------------------------
        # RETRIEVED CONTEXT
        # -------------------------------------------------

        with st.expander(
            "Retrieved context"
        ):

            for item in result["retrieved"]:

                st.write(
                    f"**[{item['modality']}]** "
                    f"{item['source']} — "
                    f"page {item['page']} "
                    f"(score {item['score']:.3f})"
                )

    # -----------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # -----------------------------------------------------

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "images": image_paths,
        }
    )