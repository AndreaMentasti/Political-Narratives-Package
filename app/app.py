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

# ───────────────────────── Constants ─────────────────────────
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
PAPER_PATH = "data/paper.pdf"

FIXED_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 400
DEFAULT_LOCAL_MODEL = "llama3.2:3b"

# --- robust truthy parsing ---
def _truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "on")

def _get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

ALLOW_LOCAL = _truthy(os.environ.get("ALLOW_LOCAL")) or _truthy(_get_secret("ALLOW_LOCAL"))

# ───────────────────────── Page header ─────────────────────────
st.set_page_config(page_title="Political Narratives — Guide + Paper Q&A", layout="wide")
st.title("Political Narratives — Guide + Paper Q&A")

# ───────────────────────── Sidebar ─────────────────────────
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

# ───────────────────────── Helpers (Q&A) ─────────────────────────
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
    - OpenAI path only if a key is provided.
    - Otherwise warns and stops execution gracefully.
    """
    if provider.startswith("OpenAI"):
        if not user_key:
            st.warning("Paste your OpenAI API key in the sidebar to ask questions.")
            st.stop()
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=FIXED_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            api_key=user_key,
        )

    if ChatOllama is None:
        st.error("Ollama not available in this environment.")
        st.stop()
    return ChatOllama(
        model=local_model,
        temperature=FIXED_TEMPERATURE,
        model_kwargs={"num_predict": 256, "num_ctx": 2048, "top_k": 30, "top_p": 0.9},
    )

STRICT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a careful research assistant. Answer the user's question using mainly the information in the provided paper excerpts.\n"
        "If the excerpts do not contain the answer, look carefully again, and then you can refer to external knowledge to give an answer.\n"
        "Excerpts from the paper:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# ───────────────────────── Helpers (Guide tab) ─────────────────────────
def _init_guide_state():
    # notes: per-step free text
    # done:     which checkboxes are ticked
    # registry: all checkbox keys that exist (for per-step totals)
    st.session_state.setdefault("guide", {
        "current_step": 1,
        "notes": {1: "", 2: "", 3: "", 4: "", 5: ""},
        "done": {},
        "registry": {}  # e.g., {"s1_scope_q1": True, "s1_scope_q2": True, ...}
    })

def question_card(title: str, how_to: list[str], ask_yourself: list[str], key_prefix: str):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if how_to:
            st.markdown("**How to approach**")
            st.markdown("\n".join([f"- {p}" for p in how_to]))
        if ask_yourself:
            st.markdown("**Ask yourself**")
            for i, q in enumerate(ask_yourself, start=1):
                cb_key = f"{key_prefix}_q{i}"
                # Register for totals
                st.session_state["guide"]["registry"][cb_key] = True
                # Render and store value
                checked = st.checkbox(
                    q,
                    key=cb_key,
                    value=st.session_state["guide"]["done"].get(cb_key, False)
                )
                st.session_state["guide"]["done"][cb_key] = checked

def render_step(step: int):
    """
    Content for each of the 5 steps from your paper:
    1) Select/define topic
    2) Identify source & extract data
    3) Identify relevant characters
    4) Prepare prompt(s)
    5) Obtain predictions & assemble outputs
    """
    # --- STEP 1 ---
    if step == 1:
        st.subheader("Step 1 — Select and define the topic")
        st.caption("A precise topic definition anchors character selection and downstream analysis.")
        question_card(
            "Define a clear topic",
            how_to=[
                "A well-defined topic is a prerequisite for fruitful narrative analysis. The clearer the topic, the more straightforward the identification of relevant characters and the exploration of the research question. Topic choice should weigh the research question, data availability, and available resources, while balancing specificity vs. generalizability. Over-narrow topics risk too few characters or narratives; over-broad topics make it difficult to restrict analysis to a manageable set. Make explicit what is in and what is out.,
                "Example: In Gehring & Grigoletto (2025) we analyze the political economy of climate change. From the literature we identify two dominant discussions—scientific evidence and policy responses—and, given our focus on political economy, we restrict attention to policy narratives, excluding debates on the scientific reality and predictability of climate change."
            ],
            ask_yourself=[
                "Is the topic expressed in 1–2 sentences?",
                "Did I specify geography, time window, and mediums?",
                "Which narrative lens (responsibility, risk, justice, solutions) am I foregrounding?",
                "What are my inclusion/exclusion rules and initial keywords/entities?"
            ],
            key_prefix="s1_scope"
        )
        question_card(
            "Why this topic now?",
            how_to=[
                "Connect to a decision, event, or policy window.",
                "Name stakeholders who care about this analysis."
            ],
            ask_yourself=[
                "Who will use the answer and for what?",
                "What would make this analysis timely and useful?"
            ],
            key_prefix="s1_rationale"
        )
        st.text_area("Annotations for Step 1 (optional)", key="notes_s1",
                     value=st.session_state["guide"]["notes"][1], height=120)
        st.session_state["guide"]["notes"][1] = st.session_state["notes_s1"]

    # --- STEP 2 ---
    if step == 2:
        st.subheader("Step 2 — Identify the source and extract data")
        st.caption("Choose sources (e.g., newspapers, social media, transcribed TV/radio/YouTube, surveys).")
        question_card(
            "Source selection",
            how_to=[
                "List candidate sources and justify their relevance.",
                "Decide access path (APIs, archives, scraping, existing corpora)."
            ],
            ask_yourself=[
                "Do my sources cover the populations/mediums defined in Step 1?",
                "Is coverage balanced across time and outlets?"
            ],
            key_prefix="s2_sources"
        )
        question_card(
            "Pre-processing plan",
            how_to=[
                "Language filtering, deduplication, optional geo-filtering.",
                "Document parsing and metadata normalization."
            ],
            ask_yourself=[
                "How will I de-duplicate and filter language?",
                "What metadata (date, outlet, geo) will I retain?"
            ],
            key_prefix="s2_preproc"
        )
        st.text_area("Annotations for Step 2 (optional)", key="notes_s2",
                     value=st.session_state["guide"]["notes"][2], height=120)
        st.session_state["guide"]["notes"][2] = st.session_state["notes_s2"]

    # --- STEP 3 ---
    if step == 3:
        st.subheader("Step 3 — Identify relevant characters")
        st.caption("Map the topic into human and instrument actors with agency and claims.")
        question_card(
            "Character buckets & exemplars",
            how_to=[
                "Aim for 4–6 buckets (government, industry, NGOs, experts, citizens).",
                "List named exemplars per bucket (ministries, firms, unions, NGOs)."
            ],
            ask_yourself=[
                "Which buckets matter most and why?",
                "Who are 2–5 named exemplars per bucket?"
            ],
            key_prefix="s3_buckets"
        )
        question_card(
            "Claims, omissions, and proxies",
            how_to=[
                "Write 1–2 typical claims per bucket.",
                "Note missing/silenced voices and how to surface them.",
                "Plan proxies to detect actors (NER lists, regex, acronyms, roles)."
            ],
            ask_yourself=[
                "What are the usual claims each bucket makes?",
                "Who is missing but should be visible?",
                "What proxies will I use to detect these actors in text?"
            ],
            key_prefix="s3_claims"
        )
        st.text_area("Annotations for Step 3 (optional)", key="notes_s3",
                     value=st.session_state["guide"]["notes"][3], height=120)
        st.session_state["guide"]["notes"][3] = st.session_state["notes_s3"]

    # --- STEP 4 ---
    if step == 4:
        st.subheader("Step 4 — Prepare the prompt(s)")
        st.caption("Specify the mapping from raw text to (M, R) with a simple, consistent schema.")
        question_card(
            "Task & input unit",
            how_to=[
                "Choose ONE main task: classify / extract / summarize / compare / generate.",
                "Pick the text unit: headline/lead/paragraph/full article (speed vs. context)."
            ],
            ask_yourself=[
                "Is the task singular and clear?",
                "Is the text unit appropriate for the task?"
            ],
            key_prefix="s4_task"
        )
        question_card(
            "Output schema & constraints",
            how_to=[
                "Define a JSON schema (keys, allowed labels, rationale).",
                "Add guardrails: cite spans, no external knowledge, be concise."
            ],
            ask_yourself=[
                "Is the schema unambiguous and machine-readable?",
                "Do I include 2–4 worked examples (few-shots), including tricky cases?"
            ],
            key_prefix="s4_schema"
        )
        st.text_area("Annotations for Step 4 (optional)", key="notes_s4",
                     value=st.session_state["guide"]["notes"][4], height=120)
        st.session_state["guide"]["notes"][4] = st.session_state["notes_s4"]

    # --- STEP 5 ---
    if step == 5:
        st.subheader("Step 5 — Obtain predictions and assemble outputs")
        st.caption("Run annotation, parse JSON, and build tidy outputs (stage flags, presence, role dummies).")
        question_card(
            "Annotation & storage",
            how_to=[
                "Decide batch size, retries, and timeouts.",
                "Store JSONL/CSV with: doc_id, span_id, label, rationale, annotator, timestamp."
            ],
            ask_yourself=[
                "What pilot size and QC checks will I run first?",
                "Where do parsed outputs live and how will I version them?"
            ],
            key_prefix="s5_annot"
        )
        question_card(
            "Quality checks & assembly",
            how_to=[
                "Compute agreement/self-consistency; add gold examples.",
                "Assemble tidy panel with stage-1 flags, presence m_i,k, and role dummies r_i,k."
            ],
            ask_yourself=[
                "What metrics signal acceptable quality?",
                "How do I visualize or audit the outputs?"
            ],
            key_prefix="s5_qc"
        )
        st.text_area("Annotations for Step 5 (optional)", key="notes_s5",
                     value=st.session_state["guide"]["notes"][5], height=120)
        st.session_state["guide"]["notes"][5] = st.session_state["notes_s5"]

def render_guide_tab():
    _init_guide_state()
    st.markdown("Use this walkthrough to plan your pipeline. Nothing is mandatory; check items you’ve considered and jot notes.")

    # Step selector (free navigation; no gating)
    step = st.segmented_control(
        "Steps",
        options=[1, 2, 3, 4, 5],
        format_func=lambda i: {1:"1 • Topic", 2:"2 • Data", 3:"3 • Characters", 4:"4 • Prompts", 5:"5 • Outputs"}[i],
        key="guide_step_selector"
    )
    st.session_state["guide"]["current_step"] = step

    # Mini progress in the sidebar
    with st.sidebar:
        st.divider()
        st.markdown("## Guide progress")

        done = st.session_state["guide"]["done"]
        registry = st.session_state["guide"]["registry"]

        def _step_complete(prefix: str) -> bool:
            keys = [k for k in registry.keys() if k.startswith(prefix)]
            return len(keys) > 0 and all(done.get(k, False) for k in keys)

        st.write(f"{'✅' if _step_complete('s1_') else '⬜️'} Step 1 — Topic")
        st.write(f"{'✅' if _step_complete('s2_') else '⬜️'} Step 2 — Data")
        st.write(f"{'✅' if _step_complete('s3_') else '⬜️'} Step 3 — Characters")
        st.write(f"{'✅' if _step_complete('s4_') else '⬜️'} Step 4 — Prompts")
        st.write(f"{'✅' if _step_complete('s5_') else '⬜️'} Step 5 — Outputs")
        st.caption("A step turns green only when all its checkboxes are marked.")

    render_step(step)

# ───────────────────────── Tabs ─────────────────────────
tab_guide, tab_qa = st.tabs(["Guide (5-step pipeline)", "Paper Q&A"])

# Tab 1 — GUIDE
with tab_guide:
    render_guide_tab()

# Tab 2 — Q&A (your original flow, unchanged)
with tab_qa:
    if vs is None:
        st.info("No documents indexed. Add your paper at data/paper.pdf and rerun.")
    else:
        llm = get_llm(provider, user_key, DEFAULT_LOCAL_MODEL)

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

        st.caption("Ask questions strictly about the paper. The assistant will answer mainly from the PDF.")
        q = st.text_input("Your question", key="qa_question_input")

        if q:
            with st.spinner("Thinking..."):
                out = qa({"query": q})
            st.write(out["result"])
