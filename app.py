# app.py (fixed: handles missing docstore.json correctly)
import os
import shutil
import streamlit as st

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
# embeddings + llm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# FAISS vector store import (try both common paths)
try:
    from llama_index.vector_stores.faiss import FaissVectorStore
except Exception:
    try:
        from llama_index.vector_stores.faiss.base import FaissVectorStore
    except Exception:
        st.error("Install the FAISS vector-store package for LlamaIndex: pip install llama-index-vector-stores-faiss")
        st.stop()

import faiss

# ========== CONFIG ==========
PERSIST_DIR = "./storage"
DATA_DIR = "./data"
FAISS_PATH = os.path.join(PERSIST_DIR, "faiss.index")
DOCSTORE_JSON = os.path.join(PERSIST_DIR, "docstore.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Set up LlamaIndex Settings
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.llm = Ollama(model="llama3:8b", request_timeout=120.0)

# Optional chunking tweaks
Settings.chunk_size = 1024
Settings.chunk_overlap = 100

# ========== HELPERS ==========
def faiss_exists() -> bool:
    return os.path.exists(FAISS_PATH) and os.path.exists(DOCSTORE_JSON)


def load_faiss_index() -> faiss.Index:
    return faiss.read_index(FAISS_PATH)


def save_faiss_index(ix: faiss.Index):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    faiss.write_index(ix, FAISS_PATH)


def build_storage_context_with_faiss(faiss_index: faiss.Index, use_persist_dir_if_safe: bool = False) -> StorageContext:
    """
    Build a StorageContext with the provided Faiss index.
    If use_persist_dir_if_safe is True and docstore.json exists, we pass persist_dir to StorageContext
    so that it can use existing docstore. Otherwise we do not pass persist_dir to avoid attempted loads.
    """
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    if use_persist_dir_if_safe and os.path.exists(DOCSTORE_JSON):
        # Safe to tell StorageContext to use persist_dir because docstore actually exists
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=PERSIST_DIR,
        )
    else:
        # Avoid passing persist_dir (prevents LlamaIndex from trying to load docstore.json)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
    return storage_context


# ========== HELPERS ==========

# ... keep your other helper functions like faiss_exists, save_faiss_index etc. ...
def clear_index_on_upload():
    """Callback to clear the index and query box when new files are uploaded."""
    if "index" in st.session_state:
        del st.session_state.index

    # ✅ ADD THIS LINE to clear the query text input
    if "query_input" in st.session_state:
        st.session_state.query_input = "" 

    st.info("New file(s) uploaded. Please click 'Build Fresh Index' to update the knowledge base.")


def build_index(data_dir):
    """
    Checks for removed files, then builds a brand new index from all documents in the data_dir.
    This function deletes any old index to ensure a fresh start.
    """
    # 1. Check for removed files against the old index (if it exists)
    temp_index = load_index_if_available()
    if temp_index:
        existing_files = {doc.metadata.get("file_name") for doc in temp_index.docstore.docs.values()}
        current_files = {f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))}
        removed_files = existing_files - current_files
        
        if removed_files:
            st.warning(f"Detected that {len(removed_files)} file(s) were removed. A fresh index will be built to reflect this change.")

    # 2. Proceed with the reliable "nuke and rebuild"
    st.info("Starting fresh build...")
    
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        
    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        st.warning("No documents found in the data directory.")
        return None
    
    st.info(f"Loaded {len(documents)} documents. Now building index...")
    
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    storage_context = build_storage_context_with_faiss(faiss_index) # Your existing helper
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    save_faiss_index(index.vector_store._faiss_index) # Your existing helper
    
    st.success("✅ New index built and saved successfully.")
    return index
    

def load_index_if_available():
    """Load the index if it exists, otherwise return None."""
    if faiss_exists():
        try:
            faiss_index = load_faiss_index()
            # use_persist_dir_if_safe is now gone, we just need to load the docstore
            storage_context = StorageContext.from_defaults(
                vector_store=FaissVectorStore(faiss_index=faiss_index),
                persist_dir=PERSIST_DIR,
            )
            return load_index_from_storage(storage_context)
        except Exception as e:
            st.error(f"Error loading index: {e}. Please reset and rebuild.")
            return None
    return None


def pretty_sources(response):
    try:
        nodes = getattr(response, "source_nodes", []) or []
        if not nodes:
            st.caption("No source nodes returned.")
            return
        st.markdown("**Sources:**")
        for i, n in enumerate(nodes, start=1):
            meta = getattr(n.node, "metadata", {}) or {}
            fname = meta.get("file_name") or meta.get("filename") or meta.get("file_path") or "Unknown"
            snippet = (n.node.get_content() or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            st.write(f"{i}. *{fname}*")
            st.caption(snippet)
    except Exception:
        st.caption("Sources could not be displayed.")


# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Project Gavel (Local)", layout="wide")
st.title("⚖️ Project Gavel — Local Legal Q&A")
st.markdown("Upload PDFs, build an index, and query with a local LLM.")

with st.sidebar:
    st.header("Controls")
    st.markdown(f"**LLM:** `llama3:8b` (Ollama)")
    st.markdown(f"**Embeddings:** `{EMBED_MODEL_NAME}`")
    if st.button("🔴 Reset / Delete Index"):
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        if "index" in st.session_state:
            del st.session_state.index
        st.success("Index deleted. Upload new files to begin.")
        st.rerun()

if "saved_files" not in st.session_state:
    st.session_state.saved_files = set()

uploaded = st.file_uploader(
    "📂 Upload PDF or text files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    on_change=clear_index_on_upload
)

# --- New logic to sync the /data folder with the uploader UI ---

# Get the set of filenames currently present in the uploader widget
current_ui_files = {f.name for f in uploaded} if uploaded else set()

# 1. Handle Deletions: Find files that are in our tracker but NOT in the UI anymore
files_to_delete = st.session_state.saved_files - current_ui_files
for fname in files_to_delete:
    file_path = os.path.join(DATA_DIR, fname)
    if os.path.exists(file_path):
        os.remove(file_path)
        st.toast(f"🗑️ Removed '{fname}' from the data folder.")

# 2. Handle Additions: Find files that are in the UI but NOT in our tracker yet
files_to_add = current_ui_files - st.session_state.saved_files
if files_to_add:
    os.makedirs(DATA_DIR, exist_ok=True)
    for uf in uploaded:
        if uf.name in files_to_add:
            path = os.path.join(DATA_DIR, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
    st.success(f"✅ Saved {len(files_to_add)} new file(s) to the `/data` folder.")

# 3. Finally, update our tracker to match the current state of the UI
st.session_state.saved_files = current_ui_files


# A SINGLE, RELIABLE BUTTON TO BUILD THE INDEX FROM SCRATCH
if st.button("📖 Build Fresh Index from All Files"):
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        st.warning("⚠️ Please upload some files first.")
    else:
        with st.spinner("Building a new index from all files in ./data..."):
            # Call the new, reliable build function
            st.session_state.index = build_index(DATA_DIR)

st.divider()
st.header("Ask a Question")

query = st.text_input("Enter your legal question:", key="query_input")
if query:
    # This logic is now much simpler. It only looks for an index in the session.
    if "index" not in st.session_state:
        st.warning("⚠️ Please build the index first.")
    else:
        index = st.session_state.index
        with st.spinner(f"Thinking with llama3:8b..."):
            qe = index.as_query_engine(
                similarity_top_k=5,
                response_mode="tree_summarize"
            )
            resp = qe.query(query)
        st.subheader("Answer")
        st.write(resp.response)
        pretty_sources(resp) # Your existing helper