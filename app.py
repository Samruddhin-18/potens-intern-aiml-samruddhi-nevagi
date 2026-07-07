import time

import streamlit as st

from rag_core import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    PDF_DIRECTORY,
    answer_question_multilingual,
    build_retrieval_chain,
    build_vector_store,
    compute_docs_signature,
    get_chunk_metadata,
    is_insufficient_context,
    load_vector_store,
    save_vector_store,
)

st.set_page_config(page_title="DocLens | Document Q&A", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 960px; }

        h1, h2, h3, h4 { font-weight: 700; letter-spacing: -0.02em; }

        .doclens-hero {
            text-align: center;
            padding: 0 0 0.75rem 0;
        }
        .doclens-hero h1 {
            font-size: 2.1rem;
            margin-bottom: 0.15rem;
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .doclens-hero p {
            font-size: 1rem;
            color: rgba(120, 120, 120, 0.95);
            font-style: italic;
            margin-top: 0;
        }

        .doclens-answer {
            background-color: rgba(79, 70, 229, 0.07);
            border-left: 4px solid #7C3AED;
            padding: 1.1rem 1.3rem;
            border-radius: 0.5rem;
            margin-top: 0.5rem;
            line-height: 1.55;
        }
        .doclens-chunk {
            background-color: rgba(120, 120, 120, 0.06);
            padding: 0.8rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.6rem;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            border: none;
            font-weight: 600;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, #4338CA, #6D28D9);
        }

        .doclens-example-btn button {
            border-radius: 2rem !important;
            font-size: 0.85rem !important;
            padding: 0.25rem 0.9rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔎 DocLens AI")
    st.caption("RAG-powered document Q&A")

    st.divider()

    st.markdown("### ⚙️ Configuration")
    st.markdown(f"🧠 **LLM Model:** `{LLM_MODEL}`")
    st.markdown(f"🧬 **Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown(f"📂 **Source folder:** `{PDF_DIRECTORY}`")

    st.divider()

    st.markdown("### 📚 Knowledge Base")
    if "vectors" in st.session_state:
        st.success("Vector store ready")
    else:
        st.info("Vector store not built yet")

    build_clicked = st.button("🔄 Build / Rebuild Vector Store", use_container_width=True)

    if "kb_stats" in st.session_state:
        stats = st.session_state.kb_stats
        col1, col2 = st.columns(2)
        col1.metric("Documents", stats["total_documents"])
        col2.metric("Chunks", stats["total_chunks"])

        with st.expander(f"📁 Loaded PDFs ({len(stats['pdf_names'])})"):
            for name in stats["pdf_names"]:
                st.markdown(f"- {name}")

    st.divider()
    st.caption("Powered by Groq + LangChain + FAISS")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="doclens-hero">
        <h1>DocLens AI</h1>
        <p>"Ask. Verify. Understand."</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not GROQ_API_KEY:
    st.error("🚫 GROQ_API_KEY is missing. Please set it in your .env file.")
    st.stop()

def _activate_vector_store(vectors, stats):
    """Wire a loaded/built FAISS index into session state as the active retrieval chain."""
    st.session_state.vectors = vectors
    st.session_state.kb_stats = stats
    st.session_state.retrieval_chain = build_retrieval_chain(vectors)


# ---------------------------------------------------------------------------
# Auto-load persisted vector store on startup (skips rebuilding if unchanged)
# ---------------------------------------------------------------------------
if "vectors" not in st.session_state and not build_clicked:
    current_signature = compute_docs_signature(PDF_DIRECTORY)
    loaded = load_vector_store()
    if loaded is not None:
        saved_vectors, saved_stats, saved_signature = loaded
        if current_signature is not None and current_signature == saved_signature:
            _activate_vector_store(saved_vectors, saved_stats)
            st.toast("Loaded existing vector store from disk.", icon="📦")
        else:
            # Documents changed since the index was saved — rebuild automatically.
            with st.spinner("📄 Documents changed. Rebuilding the vector store..."):
                try:
                    vectors, stats = build_vector_store()
                    save_vector_store(vectors, stats, current_signature)
                    _activate_vector_store(vectors, stats)
                    st.toast("Documents changed — vector store rebuilt.", icon="🔄")
                except Exception as e:
                    st.error(f"❌ Failed to rebuild the vector store: {e}")

# ---------------------------------------------------------------------------
# Vector store build (triggered from sidebar)
# ---------------------------------------------------------------------------
if build_clicked:
    spinner_text = (
        "🔄 Rebuilding the vector store..."
        if "vectors" in st.session_state
        else "🔍 Loading documents and building the vector store..."
    )
    with st.spinner(spinner_text):
        try:
            vectors, stats = build_vector_store()
            signature = compute_docs_signature(PDF_DIRECTORY)
            save_vector_store(vectors, stats, signature)
            _activate_vector_store(vectors, stats)
            st.success("✅ Vector Store DB is ready!")
            st.toast("Vector store is ready!", icon="✅")
        except ValueError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Failed to build the vector store: {e}")

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "What are the early signs of autism?",
    "How is autism diagnosed?",
    "What treatment options are discussed in the documents?",
]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

with st.container(border=True):
    st.markdown("#### 💬 Ask a question")
    user_question = st.text_input(
        "What do you want to ask from the documents?",
        value=st.session_state.pending_question,
        placeholder="e.g. What are the early signs of autism?",
        label_visibility="collapsed",
    )
    ask_clicked = st.button("🚀 Ask", type="primary", use_container_width=True)

    st.caption("💡 Try one of these example questions:")
    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(example_cols, EXAMPLE_QUESTIONS):
        with col:
            st.markdown('<div class="doclens-example-btn">', unsafe_allow_html=True)
            if st.button(question, use_container_width=True, key=f"example_{question}"):
                st.session_state.pending_question = question
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

if ask_clicked and user_question:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please build the vector store from the sidebar first.")
    else:
        try:
            with st.spinner("🤖 Thinking..."):
                start = time.process_time()
                response = answer_question_multilingual(
                    st.session_state.retrieval_chain, st.session_state.vectors, user_question
                )
                elapsed = time.process_time() - start

            answer_text = response["answer"]
            is_insufficient = is_insufficient_context(response["original_answer"])

            if is_insufficient:
                st.warning("⚠️ The documents do not contain enough information to answer this question.")
            else:
                st.success("✅ Answer generated successfully!")

            with st.container(border=True):
                st.markdown("#### ✅ Answer")
                st.markdown(f'<div class="doclens-answer">{answer_text}</div>', unsafe_allow_html=True)

            num_chunks = len(response["context"])
            meta_cols = st.columns(2)
            meta_cols[0].caption(f"⏱️ Response time: {elapsed:.2f}s")
            meta_cols[1].caption(f"🧩 Chunks retrieved: {num_chunks}")

            st.markdown("#### 📎 Sources & Citations")
            if not response["context"]:
                st.info("No supporting chunks were retrieved for this question.")
            for i, doc in enumerate(response["context"], start=1):
                meta = get_chunk_metadata(doc)
                with st.expander(f"[{i}] 📄 {meta['source']} — {meta['page_label']}"):
                    st.markdown(f"**Source file:** {meta['source']}")
                    st.markdown(f"**Page:** {meta['page_label']}")
                    st.markdown(f"**Chunk ID:** {meta['chunk_id']}")
                    st.markdown(f"**Chunk length:** {meta['char_count']} characters")
                    st.markdown("**Relevant snippet:**")
                    st.markdown(f'<div class="doclens-chunk">{meta["snippet"]}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Failed to answer the question: {e}")
elif ask_clicked and not user_question:
    st.warning("⚠️ Please enter a question first.")
