import os, glob
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# LOCAL embeddings (no API)
from langchain_community.embeddings import HuggingFaceEmbeddings

# LOCAL chat via Ollama (no API)
from langchain_community.chat_models import ChatOllama

# OPTIONAL: OpenAI chat if user brings a key
from langchain_openai import ChatOpenAI

DEFAULT_LOCAL_MODEL = "llama3.2:3b"   # also try "qwen2.5:3b"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# ---- Speed-friendly defaults ----
FIXED_TEMPERATURE = 0.1
OLLAMA_KWARGS = {
    "num_predict": 256,   # cap output tokens (major speed win). Try 128 for even faster.
    "num_ctx": 2048,      # context window; keep modest
    "top_k": 30,
    "top_p": 0.9,
}
OPENAI_MAX_TOKENS = 400  # keep OpenAI short too

st.set_page_config(page_title="Political Narratives — Local Q&A & Playground", layout="wide")
st.title("Political Narratives — Local Q&A (RAG) + Prompt Playground")

# ── Sidebar (single block, unique keys) ──────────────────────────────────────
with st.sidebar:
    st.subheader("Model settings")
    provider = st.selectbox(
        "Provider",
        ["Local (Ollama)", "OpenAI (bring your own key)"],
        index=0,
        key="provider_select",
    )

    # Removed the temperature slider (fixed low temp used instead)

    local_model = DEFAULT_LOCAL_MODEL
    user_key = None

    if provider.startswith("Local"):
        st.markdown("### Local model (Ollama)")
        local_model = st.selectbox(
            "Ollama model",
            [DEFAULT_LOCAL_MODEL, "qwen2.5:3b", "llama3.2:3b"],
            index=0,
            key="ollama_model_select",
        )
        st.caption("Tip: smallest models + short answers reply fastest.")
    else:
        user_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used only in your session. Leave empty to stay local.",
            key="openai_key_input",
        )

# ---- Warmup local model so first response is not slow ----
@st.cache_resource(show_spinner=False)
def warmup_local_model(model_name: str):
    try:
        _ = ChatOllama(model=model_name, temperature=0.0).invoke("hi")
    except Exception:
        pass

if provider.startswith("Local"):
    warmup_local_model(local_model)

# ── Helpers ─────────────────────────────────────────────────────────────────
ASSISTANT_SYSTEM = (
    "You are a helpful research assistant and project coach for applying the Political Narratives framework, "
    "especially the drama triangle character recognition to political narratives. Prefer using retrieved context "
    "from the user's paper/repo when it is relevant. If the context is missing, sparse, or not directly relevant, "
    "answer from general knowledge and common sense, and be a guide to the user. Be concise and concrete: "
    "1) ask up to 2 clarifying questions if the query is broad or ambiguous; "
    "2) propose a short, step-by-step plan if the user wants to ‘do’ something; "
    "3) when appropriate, give a tiny example taken from the paper if the question doesn't require specific examples (≤5 lines)."
)

def load_pdf(path: str):
    docs = []
    if not os.path.exists(path):
        return docs
    try:
        pdf = PdfReader(path)
        for i, p in enumerate(pdf.pages, start=1):
            text = p.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(path), "page": i}))
    except Exception as e:
        st.warning(f"Could not read PDF '{path}': {e}")
    return docs

def get_llm(provider: str, user_key: str | None, local_model: str):
    """
    Return an LLM handle based on provider selection.
    - OpenAI path activates only if a key is provided (short outputs).
    - Otherwise we fall back to local Ollama with speed-friendly kwargs.
    """
    if provider.startswith("OpenAI") and user_key:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=FIXED_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            api_key=user_key,
        )
    # Local Ollama (free) with capped output for speed
    return ChatOllama(
        model=local_model,
        temperature=FIXED_TEMPERATURE,
        model_kwargs=OLLAMA_KWARGS,
    )

def load_files(patterns):
    paths, docs = [], []
    for p in patterns:
        paths.extend(glob.glob(p, recursive=True))
    for path in paths:
        if os.path.isdir(path): 
            continue
        if not path.endswith((".md", ".py", ".txt", ".yml", ".yaml", ".json")):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs.append(Document(page_content=text, metadata={"source": os.path.relpath(path)}))
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
    return docs

@st.cache_resource(show_spinner=True)
def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_docs = []

    # 1) Papers: index ALL PDFs in data/
    for pdf_path in glob.glob("data/*.pdf"):
        for d in load_pdf(pdf_path):
            for chunk in splitter.split_text(d.page_content or ""):
                all_docs.append(Document(page_content=chunk, metadata=d.metadata))

    # 2) Repo docs (optional)
    repo_docs = load_files(["README.md", "examples/**/*.md", "*.md", "scripts/**/*.py"])
    for d in repo_docs:
        for chunk in splitter.split_text(d.page_content):
            all_docs.append(Document(page_content=chunk, metadata=d.metadata))

    if not all_docs:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(all_docs, embeddings)

vs = build_vectorstore()
st.sidebar.write("Docs in index:", vs.index.ntotal if vs else 0)

def retrieve_with_scores(vs, query: str, k: int = 4):
    """
    Returns [(doc, score), ...]. For FAISS L2 distance, LOWER is better.
    If vs is None, returns [].
    """
    if vs is None:
        return []
    try:
        return vs.similarity_search_with_score(query, k=k)
    except Exception:
        docs = vs.similarity_search(query, k=k)
        return [(d, 0.0) for d in docs]

def context_is_thin(results, min_chars: int = 300):
    if not results:
        return True
    merged = "".join(d.page_content for d, _ in results)
    return len(merged.strip()) < min_chars

# ── UI Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Ask about the paper/repo (Local or OpenAI)", "Prompt playground"])

# ---------------- TAB 1: Always-helpful Q&A (simple UI) ----------------
with tab1:
    if vs is None:
        st.info("No documents indexed. Add a text-based PDF under data/ (any name) or a README.md, then Rerun.")
    else:
        llm = get_llm(provider, user_key, local_model)
        st.caption("Ask questions about the paper/repo. Uses retrieved context when available; otherwise answers from general knowledge.")
        q = st.text_input("Your question", key="qa_question_input")

        # fixed settings (simple UI)
        top_k = 4                 # fewer passages = faster
        min_context_chars = 300   # treat tiny context as thin

        if q:
            with st.spinner("Thinking..."):
                # 1) retrieve context
                results = retrieve_with_scores(vs, q, k=top_k)
                thin = context_is_thin(results, min_chars=min_context_chars)

                # 2) build context window (may be empty/thin)
                ctx = "\n\n".join(d.page_content for d, _ in results)

                # 3) always-helpful prompt (fallback allowed)
                user_prompt = (
                    "Answer the user's question. If the provided context is insufficient, "
                    "answer from general knowledge and explicitly preface with: 'Based on general knowledge'. "
                    "Use any relevant context if present.\n\n"
                    f"Question:\n{q}\n\nContext (may be partial or irrelevant):\n{ctx}"
                )

                # 4) call the model directly
                resp = llm.invoke([
                    {"role": "system", "content": ASSISTANT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ])

            # 5) render answer
            st.write(resp.content)

            # 6) show sources only if context isn't thin
            if results and not thin:
                st.markdown("**Sources:**")
                for d, _score in results:
                    src = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page", None)
                    st.code(f"{src}" + (f" (p.{page})" if page else ""))

# ---------------- TAB 2: Prompt playground ----------------
with tab2:
    st.caption("Try prompts locally with Ollama, or switch to OpenAI via the sidebar.")
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


