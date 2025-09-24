import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional local chat via Ollama (only shown locally if ALLOW_LOCAL=1)
from langchain_community.chat_models import ChatOllama

# OpenAI chat (shown online + locally if user provides a key)
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------- Constants ----------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
PAPER_PATH = "data/paper.pdf"

# Fixed, safe defaults
FIXED_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 400

# Show Ollama only if explicitly allowed (e.g., local dev: set ALLOW_LOCAL=1)
ALLOW_LOCAL = bool(os.environ.get("ALLOW_LOCAL", "")) or bool(st.secrets.get("ALLOW_LOCAL", False))
DEFAULT_LOCAL_MODEL = "llama3.2:3b"  # or "qwen2.5:3b" for speed

# ---------------- UI Header ----------------
st.set_page_config(page_title="Political Narratives — Paper Q&A & Playground", layout="wide")
st.title("Political Narratives — Paper Q&A + Prompt Playground")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Model settings")
    provider_options = ["OpenAI (bring your own key)"] if not ALLOW_LOCAL else ["OpenAI (bring your own key)", "Local (Ollama)"]
    provider = st.selectbox("Provider", provider_options, index=0, key="provider_select")

    user_key = None
    local_model = DEFAULT_LOCAL_MODEL

    if provider.startswith("OpenAI"):
        user_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Paste your key (starts with sk- or sk-proj-). Used only in your session.",
            key="openai_key_input",
        )
    else:
        # Only shown when ALLOW_LOCAL is true
        local_model = st.selectbox(
            "Ollama model",
            [DEFAULT_LOCAL_MODEL, "qwen2.5:3b", "llama3.2:3b"],
            index=0,
            key="ollama_model_select",
        )
        st.caption("Tip: smaller local models reply faster. Answers are restricted to the paper.")

# ---------------- Helpers ----------------
def load_pdf(path: str):
    docs = []
    if not os.path.exists(path):
        return docs
    try:
        pdf = PdfReader(path)
        for i, p in enumerate(pdf.pages, start=1):
            text = p.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": os.path.basename(path), "page": i}))
    except Exception as e:
        st.warning(f"Could not read PDF '{path}': {e}")
    return docs

@st.cache_resource(show_spinner=True)
def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_docs = []

    if os.path.exists(PAPER_PATH):
        for d in load_pdf(PAPER_PATH):
            for chunk in splitter.split_text(d.page_content or ""):
                all_docs.append(Document(page_content=chunk, metadata=d.metadata))
    else:
        st.warning(f"Paper not found at {PAPER_PATH}. Please add your PDF there.")

    if not all_docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(all_docs, embeddings)

vs = build_vectorstore()
st.sidebar.write("Docs in index:", vs.index.ntotal if vs else 0)

def get_llm(provider: str, user_key: str | None, local_model: str):
    """
    Returns an LLM handle:
    - OpenAI path only if a key is provided (used online; allowed locally too).
    - Local Ollama path only if ALLOW_LOCAL is True and provider is Local.
    """
    if provider.startswith("OpenAI"):
        if not user_key:
            st.warning("Paste your OpenAI API key in the sidebar to ask questions.")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=FIXED_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            api_key=user_key,
        )
    # Local (only if allowed)
    return ChatOllama(
        model=local_model,
        temperature=FIXED_TEMPERATURE,
        model_kwargs={
            "num_predict": 256,  # shorter answers = faster
            "num_ctx": 2048,
            "top_k": 30,
            "top_p": 0.9,
        },
    )

# Strict-to-paper prompt: refuse info that isn't supported by retrieved context
STRICT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a careful research assistant. Answer the user's question using ONLY the information in the provided paper excerpts.\n"
        "If the excerpts do not contain the answer, say: \"I couldn't find that in the paper.\" Do not use outside knowledge.\n\n"
        "Excerpts from the paper:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# ---------------- UI Tabs ----------------
tab1, tab2 = st.tabs(["Ask about the paper (strict)", "Prompt playground"])

# ---------------- TAB 1: Paper-only, strict Q&A ----------------
with tab1:
    if vs is None:
        st.info("No documents indexed. Add your paper at data/paper.pdf and rerun.")
    else:
        llm = get_llm(provider, user_key, local_model)

        # Build a RetrievalQA chain with our strict prompt
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": STRICT_QA_PROMPT},
            return_source_documents=False,   # Hide sources (can set True if you want to show them)
        )

        st.caption("Ask questions strictly about the paper. The assistant will answer only from the PDF.")
        q = st.text_input("Your question", key="qa_question_input")

        # Guard: on Streamlit Cloud (OpenAI only), require a key
        if provider.startswith("OpenAI") and not user_key:
            st.stop()

        if q:
            with st.spinner("Thinking..."):
                out = qa({"query": q})
            st.write(out["result"])

# ---------------- TAB 2: Prompt playground ----------------
with tab2:
    st.caption("Try prompts with OpenAI (or locally with Ollama if enabled on your machine).")
    presets = {
        "Drama Triangle annotator (concise JSON)": (
            "You are an annotator using Drama Triangle roles: Victim, Rescuer, Persecutor. "
            "Be concise. First give a one-sentence explanation; then output a JSON object with this schema only: "
            "{ 'spans': [ { 'text': <substring>, 'role': 'victim|rescuer|persecutor', 'start': int, 'end': int } ] }."
        ),
        "Explain the framework (teacher style)": (
            "Explain the Drama Triangle framework clearly for a beginner in 5 bullet points, each <= 15 words."
        ),
        "Critique & improvements": (
            "Given the text, critique the analysis using Drama Triangle concepts; suggest 3 concrete improvements."
        ),
    }

    preset_name = st.selectbox("Preset", list(presets.keys()), key="preset_select")
    system_prompt = st.text_area("System / Instruction", presets[preset_name], height=150, key="sys_prompt_area")
    user_text = st.text_area("User text (your data or example)", height=150, placeholder="Paste a paragraph or dialogue here...", key="user_text_area")
    run = st.button("Run", type="primary", key="run_playground_btn")

    if run and user_text.strip():
        llm = get_llm(provider, user_key, local_model)
        with st.spinner("Generating..."):
            resp = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ])
        st.write(resp.content)


