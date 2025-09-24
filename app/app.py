import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Optional local chat via Ollama (only if ALLOW_LOCAL is truthy)
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None  # not available / not needed online

# ---------------- Constants ----------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
PAPER_PATH = "data/paper.pdf"

FIXED_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 400
DEFAULT_LOCAL_MODEL = "llama3.2:3b"  # or "qwen2.5:3b" for speed

# --- robust truthy parsing (so "false"/"0" don't evaluate to True) ---
def _truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "on")

def _get_secret(name, default=None):
    # Safely read st.secrets without raising when no secrets file exists
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

ALLOW_LOCAL = _truthy(os.environ.get("ALLOW_LOCAL")) or _truthy(_get_secret("ALLOW_LOCAL"))

# ---------------- UI Header ----------------
st.set_page_config(page_title="Political Narratives — Paper Q&A & Playground", layout="wide")
st.title("Political Narratives — Paper Q&A + Prompt Playground")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Model settings")

    if ALLOW_LOCAL:
        provider = st.selectbox(
            "Provider",
            ["OpenAI (bring your own key)", "Local (Ollama)"],
            index=0,
            key="provider_select",
        )
    else:
        # Online default: OpenAI only
        provider = "OpenAI (bring your own key)"
        st.caption("This app uses OpenAI when you paste your API key.")

    user_key = None
    local_model = DEFAULT_LOCAL_MODEL

    if provider.startswith("OpenAI"):
        user_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Paste your key (starts with sk- or sk-proj-). Used only in your session.",
            key="openai_key_input",
        )
    elif ALLOW_LOCAL:
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
    Returns an LLM handle.
    - Online (default): OpenAI (requires key).
    - Local Ollama only if ALLOW_LOCAL is true.
    """
    if provider.startswith("OpenAI") or not ALLOW_LOCAL:
        if not user_key:
            st.warning("Paste your OpenAI API key in the sidebar to ask questions.")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=FIXED_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            api_key=user_key,
        )

    # Local (only if allowed)
    if ChatOllama is None:
        st.error("Ollama not available in this environment.")
        st.stop()
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
        llm = get_llm(provider, user_key, DEFAULT_LOCAL_MODEL)

        # Require key when using OpenAI (online default)
        if provider.startswith("OpenAI") and not user_key:
            st.stop()

        retriever = vs.as_retriever(search_kwargs={"k": 4})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": STRICT_QA_PROMPT},
            return_source_documents=False,
        )

        st.caption("Ask questions strictly about the paper. The assistant will answer only from the PDF.")
        q = st.text_input("Your question", key="qa_question_input")

        if q:
            with st.spinner("Thinking..."):
                out = qa({"query": q})
            st.write(out["result"])

# ---------------- TAB 2: Prompt playground ----------------
with tab2:
    st.caption(
        "Try prompts with OpenAI (default). "
        + ("You can also use local Ollama if enabled on your machine." if ALLOW_LOCAL else "")
    )
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
        llm = get_llm(provider, user_key, DEFAULT_LOCAL_MODEL)
        if provider.startswith("OpenAI") and not user_key:
            st.warning("Paste your OpenAI API key in the sidebar to run the playground.")
        else:
            with st.spinner("Generating..."):
                resp = llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ])
            st.write(resp.content)



