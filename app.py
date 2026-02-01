"""
====================================================================================
Purpose        : Agentic RAG Streamlit Application — Multi-Document Concept Explainer
File Name      : app.py
Author         : Vinay Kumar
Organization   : QC
Date Developed : 2025-10-27
Last Modified  : 2026-02-01
Version        : 1.0.2
Python Version : 3.12
Framework      : Streamlit
AI Frameworks  : LangChain, CrewAI, OpenAI API
Vector Store   : FAISS (In-Memory)
Embeddings     : text-embedding-3-small
LLM Model      : gpt-4o-mini
------------------------------------------------------------------------------------
Description    :
Description    :
This application enables researchers and AI practitioners to upload or link 
This application enables researchers and AI practitioners to upload or link
multiple academic PDF papers and interact with them using an Agentic RAG (Retrieval 
multiple academic PDF papers and interact with them using an Agentic RAG (Retrieval
Augmented Generation) system. It builds an intelligent multi-agent pipeline where 
Augmented Generation) system. It builds an intelligent multi-agent pipeline where
one agent retrieves relevant information across all indexed PDFs, and another agent 
one agent retrieves relevant information across all indexed PDFs, and another agent
summarizes and explains the requested concept clearly.
summarizes and explains the requested concept clearly.
The Streamlit interface provides:
Key Fixes (v1.0.1):
  - API key management (OpenAI)
  - Avoid ASCII codec crashes during PDF ingestion by normalizing/sanitizing extracted text
  - PDF upload and indexing (up to 5 documents)
  - Ensure stdout/stderr use UTF-8 (prevents unicode print/log crashes in some environments)
  - Real-time RAG-based querying and concept explanation
  - Fix clean_filename() bug (previously returned extension chars instead of the base name)
  - Chat-style responses with citation-aware answers
  - FAISS-based in-memory retrieval for fast search
  - Multi-agent reasoning powered by CrewAI and LangChain
------------------------------------------------------------------------------------
Core Components:
1. init_session_state()        : Initializes Streamlit session variables and caches.
2. multi_rag_tool()            : Cross-document retrieval using FAISS Vector Store.
3. clean_filename()            : Sanitizes filenames and prepares display titles.
4. download_and_index_pdf()    : Downloads, splits, embeds, and indexes PDF documents.
5. setup_llm_and_tools()       : Initializes OpenAI LLM and embedding models.
6. reset_app_state_callback()  : Clears the conversation and resets concept input.
7. main_app()                  : Defines the Streamlit UI flow and orchestrates agents.
------------------------------------------------------------------------------------
Usage Example:
Run this app with Streamlit CLI:
    $ streamlit run app.py
------------------------------------------------------------------------------------
Dependencies:See requirements.txt
------------------------------------------------------------------------------------
Future Enhancements:
    - Add live search option
====================================================================================
Key Fixes (v1.0.2):
  1) PDF Unicode/ASCII crash fix:
     - Normalize & sanitize extracted PDF text (e.g., →) before chunking/embedding.
     - Force stdout/stderr to UTF-8 in some runtimes.
  2) CrewAI ImportError fix:
     - Use CrewAI-native LLM (crewai.LLM) for Agent(..., llm=...)
     - Keep LangChain only for embeddings + FAISS.
  3) clean_filename() bug fix:
     - Previously returned extension chars; now returns a readable base filename.
====================================================================================
"""

# -----------------------
# Top level imports
# -----------------------
import os
import re
import sys
import uuid
import tempfile
import unicodedata
import requests

import streamlit as st

# LangChain (PDF loading, chunking, embeddings, FAISS)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# CrewAI (agents/tools/tasks)
from crewai import Crew, Task, Agent, LLM
from crewai.tools import tool


# -----------------------
# Console encoding safety (helps on Streamlit Cloud / some terminals)
# -----------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# -----------------------
# 0) Text normalization / sanitization (prevents ASCII codec errors)
# -----------------------
def normalize_pdf_text(t: str) -> str:
    """
    Normalize and sanitize PDF-extracted text to avoid Unicode/ASCII codec issues.
    Keeps content readable while replacing some common problematic glyphs.
    """
    if not t:
        return ""

    t = unicodedata.normalize("NFKC", t)

    # Replace common PDF typography / symbols that often show up
    t = (
        t.replace("\u2192", "->")   # →
         .replace("\u2190", "<-")   # ←
         .replace("\u2013", "-")    # –
         .replace("\u2014", "-")    # —
         .replace("\u00a0", " ")    # non-breaking space
    )

    # Remove non-printable ASCII control chars (keep \n and \t)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", t)
    return t


# -----------------------
# 1) Streamlit Session State Initialization
# -----------------------
def init_session_state():
    """Initializes all necessary session state variables."""
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {
            "openai": os.environ.get("OPENAI_API_KEY", ""),
        }

    if "user_name" not in st.session_state:
        st.session_state.user_name = "Researcher"

    if "indexed_docs" not in st.session_state:
        # Stores {'id': str, 'name': str, 'path': str, 'tool': FAISSInstance}
        st.session_state.indexed_docs = []

    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

    if "max_docs" not in st.session_state:
        st.session_state.max_docs = 5

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # CrewAI native LLM
    if "crewai_llm" not in st.session_state:
        st.session_state.crewai_llm = None

    # LangChain embedder (for FAISS)
    if "embedder" not in st.session_state:
        st.session_state.embedder = None

    if "uploaded_files_cache" not in st.session_state:
        st.session_state.uploaded_files_cache = []

    if "concept_input_main" not in st.session_state:
        st.session_state.concept_input_main = ""


# -----------------------
# 2) CrewAI Tool Definitions
# -----------------------
@tool("Multi-Document RAG Search")
def multi_rag_tool(question: str) -> str:
    """
    Searches across all currently indexed documents (In-Memory FAISS Vector Stores)
    for relevant information based on the user's query.
    """
    results = []
    indexed_docs = st.session_state.indexed_docs

    if not indexed_docs:
        return "No documents indexed. Please upload a PDF or enter a URL first."

    for doc in indexed_docs:
        vector_store_instance = doc["tool"]
        doc_name = doc["name"]

        try:
            retrieved_docs = vector_store_instance.similarity_search(query=question, k=4)

            if retrieved_docs:
                for retrieved_doc in retrieved_docs:
                    content = retrieved_doc.page_content
                    page = retrieved_doc.metadata.get("page", "N/A")
                    results.append(
                        f"--- SOURCE: {doc_name} (Page {page}) ---\n"
                        f"{content}\n"
                        f"--- END SOURCE ---"
                    )

        except Exception as e:
            # Keep logging unicode-safe
            print(f"Error querying {doc_name} from FAISS/VectorStore: {repr(e)}")

    if not results:
        return "Internal RAG search yielded no relevant information across all indexed documents."

    return "\n\n".join(results)


# -----------------------
# 3) Utility Functions
# -----------------------
def clean_filename(filename: str) -> str:
    """Cleans up filenames for better display titles."""
    base = os.path.splitext(os.path.basename(filename))[0]
    base = base.replace("_", " ").replace("-", " ").strip()
    return (base[:40] + "…") if len(base) > 40 else (base or "Document")


def setup_llm_and_tools():
    """
    Sets up:
      - CrewAI native LLM (for Agent llm=...)
      - LangChain embedder (for FAISS indexing)
    """
    openai_key = st.session_state.api_keys.get("openai", "").strip()

    if not openai_key:
        st.error("Please enter your OpenAI API Key to initialize the LLM and Embedder.")
        return None

    # Ensure provider libs can see the key (important on Streamlit Cloud)
    os.environ["OPENAI_API_KEY"] = openai_key

    try:
        # CrewAI native LLM (Fix for the ImportError you saw)
        crewai_llm = LLM(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=openai_key
        )
        st.session_state.crewai_llm = crewai_llm

        # LangChain embedder for FAISS
        embedder = OpenAIEmbeddings(
            api_key=openai_key,
            model="text-embedding-3-small"
        )
        st.session_state.embedder = embedder

        return crewai_llm

    except Exception as e:
        st.session_state.crewai_llm = None
        st.session_state.embedder = None
        st.error(f"Failed to initialize LLM/Embedder. Check your API key. Error: {e}")
        return None


def download_and_index_pdf(url_or_file, is_url: bool = False):
    """Downloads or saves a PDF and indexes it using LangChain/FAISS."""
    openai_key = st.session_state.api_keys.get("openai", "").strip()
    embedder = st.session_state.get("embedder")

    if not openai_key or not embedder:
        st.error("Cannot index document: Please click 'Initialize Agents and Tools' first.")
        return

    if len(st.session_state.indexed_docs) >= st.session_state.max_docs:
        st.error(f"Cannot upload more than {st.session_state.max_docs} documents.")
        return

    temp_filepath = os.path.join(st.session_state.temp_dir, f"{uuid.uuid4()}.pdf")

    try:
        # --- File download/save ---
        if is_url:
            st.info(f"Downloading PDF from: {url_or_file}...")
            response = requests.get(url_or_file, stream=True, timeout=30)
            response.raise_for_status()
            with open(temp_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_name_candidate = url_or_file.split("/")[-1].split("?")[0]
            file_name = clean_filename(file_name_candidate) if len(file_name_candidate) > 5 else "URL Document"

        else:
            st.info(f"Saving uploaded file: {url_or_file.name}...")
            with open(temp_filepath, "wb") as f:
                f.write(url_or_file.getbuffer())
            file_name = clean_filename(url_or_file.name)

        doc_index = len(st.session_state.indexed_docs) + 1
        display_name = f"Paper {doc_index} - {file_name}"

        # --- Indexing (FAISS in-memory) ---
        with st.spinner(f"Indexing '{display_name}' with In-Memory Vector Store (FAISS)..."):
            loader = PyPDFLoader(temp_filepath)
            documents = loader.load()

            # Fix: normalize/sanitize PDF text to avoid ascii/unicode codec errors downstream
            for d in documents:
                d.page_content = normalize_pdf_text(d.page_content)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(documents)

            faiss_instance = FAISS.from_documents(documents=chunks, embedding=embedder)

            st.session_state.indexed_docs.append({
                "id": str(uuid.uuid4()),
                "name": display_name,
                "path": temp_filepath,
                "tool": faiss_instance
            })

        st.success(f"Successfully indexed: **{display_name}** using In-Memory Vector Store (FAISS).")

    except Exception as e:
        st.error(f"Failed to index document. Check URL, file format, or API key. Error: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


def reset_app_state_callback():
    """Resets chat history and clears the concept input value."""
    st.session_state.chat_history = []
    st.session_state.concept_input_main = ""


# -----------------------
# 4) Main Streamlit Logic
# -----------------------
def main_app():
    init_session_state()

    st.set_page_config(layout="wide", page_title="Concept Explainer")
    st.title("📚 Agentic RAG Streamlit Application — Multi-Document Concept Explainer")

    col_upload, col_chat = st.columns([1, 2])

    # -----------------------
    # Left column (Setup + Docs)
    # -----------------------
    with col_upload:
        st.header("1. Setup & Documents")

        st.session_state.user_name = st.text_input("Hi! What's your name?", value=st.session_state.user_name)

        st.subheader("OpenAI API Key (Required), we do not store it beyond the current session")
        st.session_state.api_keys["openai"] = st.text_input(
            "OpenAI API Key (for LLM)",
            type="password",
            value=st.session_state.api_keys.get("openai", ""),
            help="Used for agent reasoning (CrewAI) and embeddings (LangChain)."
        )

        is_initialized = st.session_state.get("crewai_llm") is not None and st.session_state.get("embedder") is not None

        if is_initialized:
            button_text = "Agents and Tools INITIALIZED"
            button_type = "primary"
        else:
            button_text = "Initialize Agents and Tools"
            button_type = "secondary"

        if st.button(button_text, key="init_button", type=button_type, disabled=is_initialized):
            crewai_llm = setup_llm_and_tools()
            if crewai_llm:
                st.success("✅ LLM (CrewAI) and Embedder (LangChain) initialized. Ready for RAG.")
                st.rerun()

        if is_initialized:
            st.success("✅ **STATUS: LLM + Embedder are ready.**")
        else:
            st.info("Status: Waiting for initialization.")

        st.subheader(f"2. Index Documents ({len(st.session_state.indexed_docs)}/{st.session_state.max_docs})")

        if len(st.session_state.indexed_docs) < st.session_state.max_docs and is_initialized:
            uploaded_files = st.file_uploader(
                "Upload PDF files (drag & drop, up to 5 total)",
                type=["pdf"],
                accept_multiple_files=True
            )

            if uploaded_files:
                cached_names = [f.name for f in st.session_state.uploaded_files_cache]
                new_files = [f for f in uploaded_files if f.name not in cached_names]

                for f in new_files:
                    download_and_index_pdf(f, is_url=False)

                st.session_state.uploaded_files_cache = uploaded_files

                if new_files:
                    st.rerun()

            with st.form("url_form", clear_on_submit=True):
                url_input = st.text_input("Or submit a PDF URL (one at a time)", key="url_input_form")
                submitted_url = st.form_submit_button("Submit URL")
                if submitted_url and url_input:
                    download_and_index_pdf(url_input, is_url=True)
                    st.rerun()

        elif not is_initialized:
            st.info("Initialize Agents and Tools first to enable document upload.")
        else:
            st.info(f"Maximum of {st.session_state.max_docs} documents indexed. Delete files to add new ones.")

        st.markdown("---")
        st.subheader("Indexed Documents")
        if st.session_state.indexed_docs:
            for doc in st.session_state.indexed_docs:
                st.markdown(f"- :white_check_mark: `{doc['name']}`")
        else:
            st.info("No documents indexed yet.")

        # Debug status
        st.caption("--- DEBUG STATUS (Internal Use) ---")
        st.json({
            "initialized": is_initialized,
            "indexed_docs_count": len(st.session_state.indexed_docs),
            "concept_input_current_state": st.session_state.concept_input_main.strip()
        })
        st.caption("-----------------------------------")

    # -----------------------
    # Right column (Chat)
    # -----------------------
    with col_chat:
        st.header(f"3. Concept Chat with {st.session_state.user_name}")

        if not is_initialized:
            st.warning("Prerequisite: Please **Initialize Agents and Tools** on the left first.")
        elif not st.session_state.indexed_docs:
            st.warning("Prerequisite: Please **Index at least one Document** first.")

        ready_to_search = bool(is_initialized and st.session_state.indexed_docs)

        # Search form (only if initialized)
        if is_initialized:
            with st.form("concept_search_form", clear_on_submit=True):
                st.text_input(
                    "Enter a concept to search and explain using the indexed documents:",
                    key="concept_input_main"
                )

                submitted = st.form_submit_button(
                    "Search Indexed Documents (RAG)",
                    disabled=not ready_to_search,
                    use_container_width=True
                )

            if submitted:
                concept = st.session_state.concept_input_main.strip()
                if not concept:
                    st.error("Please enter a concept (e.g., 'self attention', 'transformer model') to search.")
                    return

                # Use cached CrewAI LLM; initialize if missing
                crewai_llm = st.session_state.get("crewai_llm") or setup_llm_and_tools()
                if not crewai_llm:
                    return

                # Agents MUST receive CrewAI-native LLM (not LangChain ChatOpenAI)
                retriever_agent = Agent(
                    role="Document Retriever",
                    goal="Execute RAG search against indexed documents (FAISS) and gather raw content.",
                    backstory=(
                        "You are a search specialist for academic papers. You must use the `multi_rag_tool` "
                        "to search indexed PDFs. Output must include raw chunks with paper name + page number."
                    ),
                    verbose=False,
                    allow_delegation=False,
                    llm=crewai_llm,
                    tools=[multi_rag_tool]
                )

                summarizer_agent = Agent(
                    role="Concept Explainer and Summarizer",
                    goal="Synthesize retrieved content into a clear, concise explanation with sources.",
                    backstory=(
                        "You are a researcher-tutor. You take raw RAG chunks, remove redundancy, and explain "
                        "the concept clearly. Always include 'Sources Used' with paper names and page numbers."
                    ),
                    verbose=False,
                    allow_delegation=False,
                    llm=crewai_llm
                )

                retriever_task = Task(
                    description=(
                        f"Execute RAG search for the concept: '{concept}' using the `multi_rag_tool`. "
                        "Return raw search results including source names and page numbers."
                    ),
                    expected_output="Raw sourced chunks from indexed PDFs relevant to the concept.",
                    agent=retriever_agent
                )

                summarizer_task = Task(
                    description=(
                        f"Using the raw results, explain the concept '{concept}' clearly. "
                        "Provide a structured answer and end with 'Sources Used:' listing all papers + page numbers."
                    ),
                    expected_output=(
                        "A clear explanation + bullet points + 'Sources Used:' section with paper/page citations."
                    ),
                    agent=summarizer_agent,
                    context=[retriever_task]
                )

                execution_crew = Crew(
                    agents=[retriever_agent, summarizer_agent],
                    tasks=[retriever_task, summarizer_task],
                    verbose=1
                )

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"**Concept Search:** {concept} (Strategy: Internal RAG Only)"
                })

                with st.spinner("Running Agentic System via RAG..."):
                    try:
                        result = execution_crew.kickoff(inputs={
                            "concept": concept,
                            "concept_input_main": concept
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": str(result)
                        })
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**Error:** Crew execution failed. Error message: `{e}`"
                        })

                st.rerun()

        # Conversation display
        st.subheader("Conversation")
        chat_container = st.container(height=520, border=True)

        for message in st.session_state.chat_history:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            with chat_container.chat_message(role):
                st.markdown(content)

        # Next step
        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("Start New Concept", on_click=reset_app_state_callback):
                pass
            st.info("Enter a new concept above and click 'Search Indexed Documents (RAG)'.")


# -----------------------
# Run the app
# -----------------------
if __name__ == "__main__":
    main_app()
