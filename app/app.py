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
from langchain.chains import RetrievalQA

# OPTIONAL: OpenAI chat if user brings a key
from langchain_openai import ChatOpenAI

DEFAULT_LOCAL_MODEL = "llama3.2:3b"   # also try "qwen2.5:3b"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Speed-friendly defaults
FIXED_TEMPERATURE = 0.1        # lower randomness; not about speed, just stability
OLLAMA_KWARGS = {
    "num_predict": 256,        # hard cap on output tokens (biggest speed win)
    "num_ctx": 2048,           # context window; keep modest
    "top_k": 30,
    "top_p": 0.9,
}
OPENAI_MAX_TOKENS = 500        # keep OpenAI outputs short too

ALLOW_LOCAL = bool(os.environ.get("ALLOW_LOCAL", "")) or bool(st.secrets.get("ALLOW_LOCAL", False)) ##

st.set_page_config(page_title="Political Narratives — Local Q&A & Playground", layout="wide")
st.title("Political Narratives — Local Q&A (RAG) + Prompt Playground")

# ── Sidebar (single block, unique keys) ──────────────────────────────────────
with st.sidebar:
    st.subheader("Model settings")

    # Show only OpenAI online (default). Locally, show both providers.
    provider_options = ["OpenAI (bring your own key)"] if not ALLOW_LOCAL else ["Local (Ollama)", "OpenAI (bring your own key)"]
    default_index = 0  # OpenAI first when hosted; Local first if you set ALLOW_LOCAL locally
    provider = st.selectbox("Provider", provider_options, index=default_index, key="provider_select")

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
        st.caption("Tip: smallest models reply fastest.")
    else:
        user_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Paste your key (starts with sk- or sk-proj-). Used only in your session.",
            key="openai_key_input",
        )

# ─────────────────────────────────────────────────────────────────────────────

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
    "You are a helpful research assistant and project coach for applying the Political Narratives framework, especially the drama triangle character recognition "
    "to political narratives. Prefer using retrieved context from the user's paper/repo when it is relevant. "
    "If the context is missing, sparse, or not directly relevant, answer from general knowledge and common sense, and be a guide to the user. Be concise and concrete: "
    "1) ask up to 2 clarifying questions if the query is broad or ambiguous; "
    "2) propose a short, step-by-step plan if the user wants to ‘do’ something; "
    "3) when appropriate, give a tiny example (≤5 lines)."
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
    Return an LLM based on provider.
    - OpenAI path activates only if a key is provided (short outputs).
    - Otherwise use local Ollama with speed-friendly kwargs.
    """
    if provider.startswith("OpenAI") and user_key:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=FIXED_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            api_key=user_key,
        )

    # Default: local, free (limit output length for speed)
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
@st.cache_resource(show_spinner=True)
def build_vectorstores():
    stores = {
        "sections_political": build_corpus_vectorstore("sections_political", ["data/sections/political_narratives.*"]),
        "sections_analysis":  build_corpus_vectorstore("sections_analysis",  ["data/sections/analysis.*"]),
        "faq":                build_corpus_vectorstore("faq",                ["data/faq/*.*"]),
        "guide":              build_corpus_vectorstore("guide",              ["data/guides/*.*"]),
    }
    # Drop empty ones
    return {k: v for k, v in stores.items() if v is not None}

stores = build_vectorstores()
total_chunks = sum(vs.index.ntotal for vs in stores.values()) if stores else 0
st.sidebar.write("Docs in index:", total_chunks)

def retrieve_top(vs, q: str, k: int = 6):
    try:
        return vs.similarity_search_with_score(q, k=k)
    except Exception:
        docs = vs.similarity_search(q, k=k)
        return [(d, 0.0) for d in docs]

def best_corpus_for_question(stores: dict, q: str):
    """
    Returns (corpus_name, results_list) where results_list = [(doc, score), ...].
    Heuristic routing:
      1) Try FAQ. If strong enough, use it.
      2) Try each section; keep the best.
      3) Try guide.
      4) If nothing substantial, return (None, []) to trigger general knowledge fallback.
    """
    if not stores:
        return None, []

    MIN_CHARS = 300
    candidates = []

    # 1) FAQ
    if "faq" in stores:
        faq_res = retrieve_top(stores["faq"], q, k=6)
        candidates.append(("faq", faq_res))

    # 2) Sections
    for name in ("sections_political", "sections_analysis"):
        if name in stores:
            res = retrieve_top(stores[name], q, k=6)
            candidates.append((name, res))

    # 3) Guide
    if "guide" in stores:
        guide_res = retrieve_top(stores["guide"], q, k=6)
        candidates.append(("guide", guide_res))

    def merged_len(res): return len("".join(d.page_content for d, _ in res).strip())
    candidates.sort(key=lambda t: merged_len(t[1]), reverse=True)

    if not candidates:
        return None, []
    best_name, best_res = candidates[0]
    if merged_len(best_res) < MIN_CHARS:
        return None, []
    return best_name, best_res

def load_text_file(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        return [Document(page_content=txt, metadata={"source": os.path.basename(path)})]
    except Exception as e:
        st.warning(f"Could not read text file '{path}': {e}")
        return []

def load_any(path: str):
    if path.lower().endswith((".md", ".txt")):
        return load_text_file(path)
    elif path.lower().endswith(".pdf"):
        return load_pdf(path)
    return []

def build_corpus_vectorstore(name: str, glob_patterns: list[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    paths = []
    for p in glob_patterns:
        paths.extend(glob.glob(p, recursive=True))
    for path in paths:
        if os.path.isdir(path):
            continue
        for d in load_any(path):
            # Tag every chunk with the corpus name
            for chunk in splitter.split_text(d.page_content or ""):
                docs.append(Document(page_content=chunk, metadata={**d.metadata, "corpus": name}))
    if not docs:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# ── UI Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Ask about the paper/repo (Local or OpenAI)", "Prompt playground"])

# ---------------- TAB 1: Always-helpful Q&A (simple UI) ----------------
# ---------------- TAB 1: Always-helpful Q&A with routing + memory ----------------
with tab1:
    if not stores:
        st.info("No documents indexed. Add files under data/sections, data/faq, data/guides, then Rerun.")
    else:
        if "chat" not in st.session_state:
            st.session_state.chat = []  # [{"role":"user"/"assistant","content": "..."}]

        llm = get_llm(provider, user_key, local_model)
        st.caption("Ask anything. The assistant uses sections/FAQ/guide when helpful; otherwise answers from general knowledge.")
        q = st.text_input("Your question", key="qa_question_input")

        colA, colB = st.columns([1,1])
        with colA:
            clear = st.button("Clear chat")
        with colB:
            show_sources = st.checkbox("Show sources when available", value=True)

        if clear:
            st.session_state.chat = []

        if q:
            # ROUTE to best corpus
            corpus, results = best_corpus_for_question(stores, q)
            ctx = "\n\n".join(d.page_content for d, _ in results) if results else ""

            # MEMORY: include last few turns for context (kept short for speed)
            history = st.session_state.chat[-6:]

            # PROMPT: always helpful; fallback allowed
            user_prompt = (
                "Answer the user's question. If the provided context is insufficient, "
                "answer from general knowledge and preface with: 'Based on general knowledge'. "
                "Use any relevant context if present. Keep the answer ~120 words unless asked otherwise.\n\n"
                f"Question:\n{q}\n\n"
                f"Context from corpus '{corpus}' (may be partial or irrelevant):\n{ctx}"
            )

            msgs = [{"role": "system", "content": ASSISTANT_SYSTEM}]
            msgs.extend(history)
            msgs.append({"role": "user", "content": user_prompt})

            with st.spinner("Thinking..."):
                resp = llm.invoke(msgs)

            # Update memory
            st.session_state.chat.append({"role": "user", "content": q})
            st.session_state.chat.append({"role": "assistant", "content": resp.content})

            # Render answer
            st.write(resp.content)

            # Sources (if we actually used a corpus)
            if show_sources and results:
                st.markdown(f"**Sources (corpus: {corpus})**")
                for d, _ in results:
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

