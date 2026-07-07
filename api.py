"""FastAPI backend exposing the DocLens RAG pipeline over HTTP.

Runs independently of the Streamlit app (streamlit run app.py) — start it with:
    uvicorn api:app --reload
Both surfaces share the same pipeline logic from rag_core.py, so there is no
duplicated document loading, chunking, or retrieval-chain code.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_core import (
    GROQ_API_KEY,
    answer_question_multilingual,
    build_retrieval_chain,
    compare_documents_for_contradiction,
    get_chunk_metadata,
    get_or_build_vector_store,
    is_insufficient_context,
)

_state: dict = {"retrieval_chain": None, "vectors": None, "kb_stats": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Please set it in your .env file.")

    vectors, stats = get_or_build_vector_store()
    _state["retrieval_chain"] = build_retrieval_chain(vectors)
    _state["vectors"] = vectors
    _state["kb_stats"] = stats
    yield
    _state.clear()


app = FastAPI(
    title="DocLens API",
    description="RAG-powered document Q&A backend for the DocLens knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the document corpus.")


class Citation(BaseModel):
    source: str
    page: str
    chunk_id: str | int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    is_insufficient: bool
    citations: list[Citation]
    detected_language: str


class ContradictRequest(BaseModel):
    document_1: str = Field(..., min_length=1, description="Filename of the first document, e.g. 'Autism.pdf'.")
    document_2: str = Field(..., min_length=1, description="Filename of the second document.")
    topic: str = Field(..., min_length=1, description="The topic to compare between the two documents.")


class ContradictResponse(BaseModel):
    conflict: bool
    reasoning: str
    evidence_document_1: list[Citation]
    evidence_document_2: list[Citation]


def _to_citations(chunk_metadatas):
    return [
        Citation(source=meta["source"], page=meta["page_label"], chunk_id=meta["chunk_id"], snippet=meta["snippet"])
        for meta in chunk_metadatas
    ]


@app.get("/health")
def health():
    return {"status": "ok", "kb_stats": _state["kb_stats"]}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    retrieval_chain = _state["retrieval_chain"]
    vectors = _state["vectors"]
    if retrieval_chain is None or vectors is None:
        raise HTTPException(status_code=503, detail="Vector store is not ready yet.")

    try:
        response = answer_question_multilingual(retrieval_chain, vectors, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer the question: {e}")

    citations = _to_citations(get_chunk_metadata(doc) for doc in response["context"])

    return AskResponse(
        answer=response["answer"],
        is_insufficient=is_insufficient_context(response["original_answer"]),
        citations=citations,
        detected_language=response["source_language"],
    )


@app.post("/contradict", response_model=ContradictResponse)
def contradict(request: ContradictRequest):
    vectors = _state["vectors"]
    if vectors is None:
        raise HTTPException(status_code=503, detail="Vector store is not ready yet.")

    try:
        result = compare_documents_for_contradiction(
            vectors, request.document_1, request.document_2, request.topic
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare documents: {e}")

    return ContradictResponse(
        conflict=result["conflict"],
        reasoning=result["reasoning"],
        evidence_document_1=_to_citations(result["evidence_document_1"]),
        evidence_document_2=_to_citations(result["evidence_document_2"]),
    )
