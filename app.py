import os
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PDF_DIRECTORY = "./autism"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="Document Q&A")
st.title("Gemma Model Document Q&A")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Please set it in your .env file.")
    st.stop()

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>

    Questions: {input}
    """
)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL)


def build_vector_store():
    """Load PDFs, split them into chunks, and embed them into a FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No PDF documents found in '{PDF_DIRECTORY}'.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors


prompt1 = st.text_input("What you want to ask from the documents?")

if st.button("Creating Vector Store"):
    if "vectors" not in st.session_state:
        with st.spinner("Loading documents and building the vector store..."):
            try:
                st.session_state.vectors = build_vector_store()
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Failed to build the vector store: {e}")
    if "vectors" in st.session_state:
        st.success("Vector Store DB is ready")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Creating Vector Store' first.")
    else:
        try:
            llm = get_llm()
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt1})
            elapsed = time.process_time() - start

            st.write(response["answer"])
            st.caption(f"Response time: {elapsed:.2f}s")

            with st.expander("Document Similarity Search"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"Failed to answer the question: {e}")
