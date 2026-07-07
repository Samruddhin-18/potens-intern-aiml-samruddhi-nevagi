# DocLens

DocLens is a Retrieval-Augmented Generation (RAG) system for asking questions over a
collection of PDF documents. It retrieves the most relevant passages from a local
knowledge base, generates a grounded answer with an LLM, and returns the exact source
snippets that support that answer. The project ships both a Streamlit UI and a FastAPI
backend on top of one shared pipeline.

## Overview

Given a folder of PDFs (currently a set of autism-related medical documents/fact sheets),
DocLens builds a searchable vector index, then lets a user ask natural-language questions
against it. Every answer is traceable back to the exact document, page, and chunk it came
from, and the system is designed to say "I don't know" rather than fabricate an answer when
the documents don't cover the question.

## Features

- **Document Q&A** — ask a question, get an answer grounded in the PDF corpus with inline citations.
- **Citations on every answer** — source filename, page number (or chunk ID as fallback), and the exact supporting text snippet.
- **Hallucination guard** — a retrieval-confidence check runs before generation; if nothing relevant is found, the LLM is never invoked.
- **Multilingual support** — ask in any language; the question is translated for retrieval and the answer is translated back, while citations stay in the source document's original language.
- **Cross-document contradiction detection** — compare how two documents discuss the same topic and get an LLM-reasoned verdict on whether they conflict.
- **Persistent vector store** — the FAISS index is cached to disk and only rebuilt when the source PDFs change.
- **Dual interface** — a Streamlit UI for interactive use and a FastAPI backend for programmatic/service access, both built on the same pipeline.

## Architecture

```
                        ┌─────────────────┐
                        │   PDF documents   │
                        │     (./autism)    │
                        └────────┬─────────┘
                                 │ load + chunk
                                 ▼
                        ┌─────────────────┐
                        │  rag_core.py      │  ← shared pipeline
                        │  - embeddings      │
                        │  - FAISS store     │
                        │  - retrieval chain │
                        │  - hallucination   │
                        │    guard           │
                        │  - translation     │
                        │  - contradiction   │
                        │    detection       │
                        └───┬───────────┬───┘
                            │           │
                 ┌──────────┘           └──────────┐
                 ▼                                 ▼
        ┌─────────────────┐               ┌─────────────────┐
        │    app.py         │               │    api.py         │
        │  Streamlit UI      │               │  FastAPI backend   │
        └─────────────────┘               └─────────────────┘
```

`rag_core.py` is the single source of truth for document loading, chunking, embedding,
retrieval, answer generation, citation formatting, translation, and contradiction analysis.
Neither `app.py` nor `api.py` duplicates that logic — they only orchestrate it for their
respective interface.

## Tech Stack

| Layer            | Technology                                            |
|-------------------|--------------------------------------------------------|
| LLM               | Groq (`openai/gpt-oss-120b`) via `langchain-groq`       |
| Embeddings        | `BAAI/bge-small-en-v1.5` via `langchain-huggingface`    |
| Vector store      | FAISS (`faiss-cpu`)                                     |
| Orchestration     | LangChain (`langchain`, `langchain-classic`)            |
| PDF parsing       | `pypdf` / `PyPDF2` via `PyPDFDirectoryLoader`           |
| Language detection| `langdetect`                                            |
| Web UI            | Streamlit                                               |
| API               | FastAPI + Uvicorn + Pydantic                            |

## Folder Structure

```
doclens/
├── app.py              # Streamlit UI
├── api.py              # FastAPI backend (/ask, /contradict)
├── rag_core.py          # Shared RAG pipeline (used by both app.py and api.py)
├── autism/              # Source PDF documents (the knowledge base)
├── faiss_index/         # Persisted FAISS index + metadata (auto-generated)
├── requirements.txt      # Python dependencies
├── .env                 # GROQ_API_KEY (not committed)
└── README.md
```

## Installation and Setup

**Prerequisites:** Python 3.10+, a [Groq API key](https://console.groq.com/).

```bash
# 1. Clone the repository and enter it
cd doclens

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
echo GROQ_API_KEY=your_key_here > .env

# 5. Add source PDFs
# Place PDF files in the ./autism directory (or change PDF_DIRECTORY in rag_core.py)
```

**Run the Streamlit UI:**
```bash
streamlit run app.py
```

**Run the FastAPI backend:**
```bash
uvicorn api:app --reload
```
Interactive API docs are then available at `http://127.0.0.1:8000/docs`.

The first run (or any run after the PDFs change) builds and persists a FAISS index to
`./faiss_index`; subsequent runs load it from disk instead of re-embedding everything.

## How the RAG Pipeline Works

1. **Load** — all PDFs in the source folder are loaded and split into per-page documents (`PyPDFDirectoryLoader`).
2. **Chunk** — pages are split into overlapping chunks (see chunking strategy below), each tagged with a `chunk_id`.
3. **Embed & index** — chunks are embedded with `BAAI/bge-small-en-v1.5` and stored in a FAISS index, persisted to disk with a content signature so it's only rebuilt when the PDFs actually change.
4. **Retrieve** — for a question, an MMR (Maximal Marginal Relevance) retriever pulls the top-k diverse, relevant chunks.
5. **Confidence check** — before generation, a separate scored similarity search checks whether any retrieved chunk clears a minimum relevance threshold (see Hallucination Prevention below).
6. **Generate** — the LLM answers strictly from the retrieved chunks via a "stuff documents" chain, using a prompt that forbids outside knowledge.
7. **Cite** — each source chunk used is returned alongside the answer with its filename, page/chunk locator, and exact text.
8. **(Optional) Translate** — if the original question wasn't in English, the answer is translated back into that language before being returned; citations are left in their original language.

## Chunking Strategy

Documents are split with `RecursiveCharacterTextSplitter` using **chunk size 1000 characters
with 200 characters of overlap**. This was chosen because:

- **1000 characters** is large enough to preserve a coherent unit of meaning (a paragraph or a
  few sentences of clinical/medical text) without pulling in so much surrounding content that
  the embedding becomes diluted and less discriminative.
- **200-character overlap (20%)** guards against splitting a sentence or idea exactly at a
  chunk boundary — if the answer-relevant fact spans a boundary, at least one chunk still
  captures it whole.
- Recursive splitting (rather than a fixed-width or sentence splitter) tries progressively
  smaller separators (paragraph → sentence → word), so chunks break at natural boundaries
  as often as possible instead of mid-sentence.
- Each chunk is tagged with a `chunk_id`, giving every citation a stable locator even for
  content where a page number isn't meaningful.

## API Endpoints

### `POST /ask`
Ask a question against the document corpus.

**Request**
```json
{ "question": "What are the early signs of autism?" }
```

**Response**
```json
{
  "answer": "...",
  "is_insufficient": false,
  "citations": [
    { "source": "auti_20193447.pdf", "page": "Page 3", "chunk_id": 32, "snippet": "..." }
  ],
  "detected_language": "en"
}
```

### `POST /contradict`
Compare how two documents discuss a topic and check for contradictions.

**Request**
```json
{
  "document_1": "Autism.pdf",
  "document_2": "auti_20193447.pdf",
  "topic": "prevalence of autism"
}
```

**Response**
```json
{
  "conflict": false,
  "reasoning": "...",
  "evidence_document_1": [ { "source": "...", "page": "...", "chunk_id": 2, "snippet": "..." } ],
  "evidence_document_2": [ { "source": "...", "page": "...", "chunk_id": 26, "snippet": "..." } ]
}
```

### `GET /health`
Returns readiness status and current knowledge-base stats (document/chunk counts).

## Hallucination Prevention Strategy

Two layers work together to keep answers grounded:

1. **Retrieval-confidence gate** — before the LLM ever sees the question, a scored
   similarity search checks whether any retrieved chunk clears a minimum normalized
   relevance score (`RAG_MIN_RELEVANCE_SCORE`, default `0.2`). If nothing is relevant
   enough, the pipeline **short-circuits** and returns the fixed message below without
   invoking the LLM at all — the model is never given a chance to guess:
   > "The provided documents do not contain sufficient information to answer this question."
2. **Grounding prompt constraint** — even when relevant context is passed to the LLM, the
   prompt explicitly instructs it to answer *only* from the given context, forbids
   outside knowledge or inference beyond what's stated, and requires it to fall back to the
   same fixed message if the context still isn't sufficient.

This same guard is reused identically by both the Streamlit UI and the `/ask` API endpoint,
so behavior is consistent across surfaces.

## Future Improvements

- Support additional file types (Word, HTML, plain text) beyond PDF.
- Add conversation memory / multi-turn follow-up questions.
- Add authentication and per-user rate limiting to the API.
- Add automated evaluation (retrieval precision/recall, answer faithfulness scoring).
- Add streaming responses for both the UI and API.
- Support incremental re-indexing instead of full rebuilds when only a few PDFs change.
- Configurable relevance threshold and retriever parameters via the UI/API rather than only env vars.

## AI Use Log

This project was built with Claude Code (Anthropic) as a pair-programming assistant across
the following areas:
- Scaffolding the initial RAG pipeline (chunking, FAISS indexing, retrieval chain) and Streamlit UI.
- Refactoring shared pipeline logic out of the Streamlit app into `rag_core.py` to avoid duplication when adding the FastAPI backend.
- Implementing the FastAPI `/ask` and `/contradict` endpoints.
- Adding the retrieval-confidence hallucination guard.
- Adding multilingual question/answer translation support.
- Drafting this README.

All AI-assisted code was reviewed, run, and manually verified (including live end-to-end
tests of both the Streamlit UI and the API endpoints) before being committed.

## Known Limitations

- **English-centric knowledge base**: the source documents and embedding model are English;
  non-English questions work via translation, but retrieval quality depends on how well the
  translated query matches the English source text.
- **Relevance threshold is heuristic**: `RAG_MIN_RELEVANCE_SCORE` is a fixed cutoff tuned by
  observation, not a calibrated statistical measure — it may occasionally reject a valid
  question or admit a marginal one.
- **No conversation memory**: each question is answered independently; there is no
  multi-turn context carried between questions.
- **Single local knowledge base**: the system indexes one fixed folder of PDFs; it does not
  support per-user or per-session document uploads.
- **No authentication**: the FastAPI backend has no auth/rate-limiting layer and is intended
  for local/trusted use, not public deployment as-is.
- **Contradiction detection is topic-scoped by retrieval**: it only compares the top-k
  retrieved passages per document for the given topic, not the full text of either document.
