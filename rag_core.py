"""Shared RAG pipeline: vector store management, retrieval chain, and citation
formatting. Used by both the Streamlit UI (app.py) and the FastAPI backend (api.py)
so the two surfaces never duplicate document-loading, chunking, or answer logic."""

import hashlib
import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langdetect import DetectorFactory, LangDetectException, detect
from pydantic import BaseModel, Field

DetectorFactory.seed = 0  # deterministic language detection

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PDF_DIRECTORY = "./autism"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"
FAISS_INDEX_DIR = "./faiss_index"
FAISS_INDEX_META_PATH = os.path.join(FAISS_INDEX_DIR, "doclens_meta.json")

# Minimum normalized relevance score (0-1, higher = more similar) a retrieved chunk must
# clear before the LLM is allowed to answer from it. Below this, retrieval is treated as
# having found nothing relevant, and we short-circuit instead of asking the LLM to guess.
MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.2"))

INSUFFICIENT_CONTEXT_MESSAGE = (
    "The provided documents do not contain sufficient information to answer this question."
)

PROMPT = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the information contained in the context below.
    Do not use any outside knowledge and do not guess or make up information.
    Every claim in your answer must be directly supported by the context — do not
    extrapolate, combine unrelated facts, or infer information that is not explicitly stated.

    If the context does not contain enough information to answer the question,
    respond with EXACTLY this sentence and nothing else:
    "{insufficient_context_message}"

    <context>
    {{context}}
    </context>

    Questions: {{input}}
    """.format(insufficient_context_message=INSUFFICIENT_CONTEXT_MESSAGE)
)


@lru_cache(maxsize=1)
def get_llm():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Please set it in your .env file.")
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL)


@lru_cache(maxsize=1)
def get_document_chain():
    return create_stuff_documents_chain(get_llm(), PROMPT)


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once per process and reuse it everywhere."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def compute_docs_signature(pdf_directory=PDF_DIRECTORY):
    """Fingerprint the PDF source folder (filenames + sizes + mtimes) to detect changes."""
    if not os.path.isdir(pdf_directory):
        return None

    entries = []
    for name in sorted(os.listdir(pdf_directory)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_directory, name)
        stat = os.stat(path)
        entries.append({"name": name, "size": stat.st_size, "mtime": stat.st_mtime})

    if not entries:
        return None

    return hashlib.md5(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


def save_vector_store(vectors, stats, signature):
    """Persist the FAISS index and its metadata to disk so future launches can skip rebuilding."""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectors.save_local(FAISS_INDEX_DIR)
    with open(FAISS_INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "signature": signature}, f)


def load_vector_store():
    """Load a previously saved FAISS index + stats from disk, if present and readable."""
    if not os.path.isfile(FAISS_INDEX_META_PATH):
        return None

    with open(FAISS_INDEX_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    vectors = FAISS.load_local(
        FAISS_INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True
    )
    return vectors, meta["stats"], meta["signature"]


def build_vector_store(pdf_directory=PDF_DIRECTORY):
    """Load PDFs, split them into chunks, and embed them into a FAISS index."""
    embeddings = get_embeddings()

    loader = PyPDFDirectoryLoader(pdf_directory)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No PDF documents found in '{pdf_directory}'.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    for i, chunk in enumerate(final_documents):
        chunk.metadata["chunk_id"] = i

    vectors = FAISS.from_documents(final_documents, embeddings)

    stats = {
        "pdf_names": sorted({os.path.basename(d.metadata.get("source", "Unknown")) for d in docs}),
        "total_documents": len(docs),
        "total_chunks": len(final_documents),
    }
    return vectors, stats


def get_or_build_vector_store():
    """Load the persisted vector store if it's fresh, otherwise (re)build and persist it.

    This is the single entry point both the Streamlit app and the API use to obtain a
    ready-to-query vector store without duplicating the load/rebuild decision logic.
    """
    current_signature = compute_docs_signature()
    loaded = load_vector_store()
    if loaded is not None:
        vectors, stats, saved_signature = loaded
        if current_signature is not None and current_signature == saved_signature:
            return vectors, stats

    vectors, stats = build_vector_store()
    save_vector_store(vectors, stats, current_signature)
    return vectors, stats


def build_retrieval_chain(vectors):
    """Wire a FAISS vector store into a retrieval chain (MMR retriever + document chain)."""
    retriever = vectors.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
    )
    return create_retrieval_chain(retriever, get_document_chain())


def format_page_label(doc):
    """Return a human-readable 'Source — Page N' label for a retrieved document.

    Falls back to the chunk id when no page number is available (e.g. non-paginated
    sources), so every citation always has a usable locator.
    """
    source = os.path.basename(doc.metadata.get("source", "Unknown source"))
    page = doc.metadata.get("page")
    if isinstance(page, int):
        page_label = f"Page {page + 1}"
    else:
        chunk_id = doc.metadata.get("chunk_id")
        page_label = f"Chunk {chunk_id}" if chunk_id is not None else "Page N/A"
    return source, page_label


def get_chunk_metadata(doc):
    """Return citation metadata (source, page label, chunk id, char count, snippet) for a
    retrieved document — the exact chunk of text the answer was generated from."""
    source, page_label = format_page_label(doc)
    chunk_id = doc.metadata.get("chunk_id", "N/A")
    char_count = len(doc.page_content)
    return {
        "source": source,
        "page_label": page_label,
        "chunk_id": chunk_id,
        "char_count": char_count,
        "snippet": doc.page_content,
    }


def is_insufficient_context(answer_text):
    return answer_text.strip() == INSUFFICIENT_CONTEXT_MESSAGE


def assess_context_sufficiency(vectors, question, k=4, min_relevance_score=MIN_RELEVANCE_SCORE):
    """Pre-check retrieval quality before letting the LLM see the question.

    Runs a scored similarity search directly against the vector store (independent of the
    MMR retriever used for the actual answer) and checks whether at least one chunk clears
    `min_relevance_score`. This catches the case where retrieval returns only weakly related
    chunks (or none at all) — the scenario most likely to make an LLM hallucinate an answer
    instead of admitting the documents don't cover the question.
    """
    try:
        scored_docs = vectors.similarity_search_with_relevance_scores(question, k=k)
    except Exception:
        scored_docs = []

    is_sufficient = any(score >= min_relevance_score for _, score in scored_docs)
    return is_sufficient, scored_docs


def answer_question(retrieval_chain, vectors, question):
    """Answer a question through the retrieval chain, guarded by a retrieval-confidence check.

    If no retrieved chunk is relevant enough, we return the standard insufficient-context
    message directly instead of invoking the LLM — the LLM is never given a chance to guess
    from irrelevant context. Both the Streamlit UI and the /ask API call this single function
    so the hallucination guard behaves identically on both surfaces.
    """
    is_sufficient, scored_docs = assess_context_sufficiency(vectors, question)
    if not is_sufficient:
        return {"answer": INSUFFICIENT_CONTEXT_MESSAGE, "context": []}

    return retrieval_chain.invoke({"input": question})


# ---------------------------------------------------------------------------
# Multilingual support
# ---------------------------------------------------------------------------

TRANSLATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise translator. Translate the user's text accurately, preserving "
            "its meaning, tone, and formatting. Output ONLY the translated text — no quotes, "
            "commentary, or explanation.",
        ),
        (
            "human",
            "Translate the following text into the language with ISO 639-1 code '{target_language}':\n\n{text}",
        ),
    ]
)


def detect_language(text):
    """Best-effort ISO 639-1 language detection; defaults to English on ambiguous/short input."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def translate_text(text, target_language):
    """Translate `text` into the given ISO 639-1 language code using the LLM."""
    if not text.strip():
        return text
    chain = TRANSLATION_PROMPT | get_llm()
    result = chain.invoke({"text": text, "target_language": target_language})
    return result.content.strip()


def answer_question_multilingual(retrieval_chain, vectors, question):
    """Multilingual wrapper around `answer_question`.

    The knowledge base and retrieval pipeline are English-only, so a non-English question is
    translated to English before retrieval/generation, and the resulting answer is translated
    back into the question's original language before being returned. Citations are drawn
    verbatim from the source documents and are never translated. Returns everything
    `answer_question` returns, plus `source_language` and the untranslated `original_answer`
    (needed to check for the insufficient-context sentinel after translation).
    """
    source_language = detect_language(question)

    retrieval_query = question
    if source_language != "en":
        retrieval_query = translate_text(question, "en")

    response = answer_question(retrieval_chain, vectors, retrieval_query)
    original_answer = response["answer"]

    translated_answer = original_answer
    if source_language != "en":
        translated_answer = translate_text(original_answer, source_language)

    return {
        "answer": translated_answer,
        "context": response["context"],
        "source_language": source_language,
        "original_answer": original_answer,
    }


# ---------------------------------------------------------------------------
# Cross-document contradiction detection
# ---------------------------------------------------------------------------

NO_EVIDENCE_MESSAGE = (
    "One or both documents do not contain sufficient information about this topic to compare."
)

CONTRADICTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful fact-checking assistant. You compare evidence excerpts from two "
            "documents on a given topic and decide whether they contradict each other. Base your "
            "judgment ONLY on the excerpts provided — do not use outside knowledge, and do not "
            "assume a conflict exists unless the excerpts actually disagree on a fact. If the "
            "excerpts are simply about different aspects of the topic without disagreeing, that is "
            "not a conflict.",
        ),
        (
            "human",
            "Topic: {topic}\n\n"
            "Evidence from Document 1 ({doc1_name}):\n{doc1_evidence}\n\n"
            "Evidence from Document 2 ({doc2_name}):\n{doc2_evidence}\n\n"
            "Do these two documents contradict each other regarding the topic?",
        ),
    ]
)


class ContradictionVerdict(BaseModel):
    """Structured LLM judgment on whether two evidence sets conflict."""

    conflict: bool = Field(..., description="True if the two documents contradict each other on the topic.")
    reasoning: str = Field(..., description="A concise explanation of the judgment, grounded in the evidence.")


def retrieve_document_passages(vectors, document_name, topic, k=4, fetch_k=40, min_relevance_score=MIN_RELEVANCE_SCORE):
    """Retrieve passages about `topic`, restricted to a single source document.

    Reuses the same vector store as the main Q&A pipeline (no separate per-document index):
    it filters FAISS's similarity search to chunks whose `source` metadata matches
    `document_name` (matched by full path or basename, so either form works), then keeps
    only chunks that clear `min_relevance_score` to avoid pulling in unrelated filler.
    """

    def _belongs_to_document(metadata):
        source = metadata.get("source", "")
        return source == document_name or os.path.basename(source) == document_name

    try:
        scored_docs = vectors.similarity_search_with_relevance_scores(
            topic, k=k, filter=_belongs_to_document, fetch_k=fetch_k
        )
    except Exception:
        scored_docs = []

    return [doc for doc, score in scored_docs if score >= min_relevance_score]


def compare_documents_for_contradiction(vectors, document_1, document_2, topic, k=4):
    """Retrieve topic-relevant evidence from two documents and ask the LLM whether they conflict.

    Returns a dict with `conflict`, `reasoning`, and the supporting evidence chunks (with full
    citation metadata) retrieved from each document.
    """
    doc1_passages = retrieve_document_passages(vectors, document_1, topic, k=k)
    doc2_passages = retrieve_document_passages(vectors, document_2, topic, k=k)

    if not doc1_passages or not doc2_passages:
        return {
            "conflict": False,
            "reasoning": NO_EVIDENCE_MESSAGE,
            "evidence_document_1": [get_chunk_metadata(doc) for doc in doc1_passages],
            "evidence_document_2": [get_chunk_metadata(doc) for doc in doc2_passages],
        }

    chain = CONTRADICTION_PROMPT | get_llm().with_structured_output(ContradictionVerdict)
    verdict = chain.invoke(
        {
            "topic": topic,
            "doc1_name": document_1,
            "doc1_evidence": "\n\n".join(doc.page_content for doc in doc1_passages),
            "doc2_name": document_2,
            "doc2_evidence": "\n\n".join(doc.page_content for doc in doc2_passages),
        }
    )

    return {
        "conflict": verdict.conflict,
        "reasoning": verdict.reasoning,
        "evidence_document_1": [get_chunk_metadata(doc) for doc in doc1_passages],
        "evidence_document_2": [get_chunk_metadata(doc) for doc in doc2_passages],
    }
