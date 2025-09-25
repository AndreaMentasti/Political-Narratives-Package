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
    # Initialize guide state + a mirror "current_step" we control
    st.session_state.setdefault("guide", {
        "current_step": "Intro",          # default landing page
        "notes": {1: "", 2: "", 3: "", 4: "", 5: ""},
        "done": {},
        "registry": {},
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

def example_card(title: str, example_md: str, key_prefix: str):
    """A simple markdown example block (no checkboxes)."""
    with st.container(border=True):
        st.markdown(f"**Example — {title}**")
        st.markdown(example_md)

def output_card(title: str, bullets: list[str] | None = None, body_md: str | None = None, key_prefix: str = ""):
    """Defines the expected output after the step (bulleted or markdown)."""
    with st.container(border=True):
        st.markdown(f"**Output — {title}**")
        if bullets:
            st.markdown("\n".join([f"- {b}" for b in bullets]))
        if body_md:
            st.markdown(body_md)

def render_intro():
    st.subheader(" Political Narratives guide")
    st.markdown(
        """
The purpose of a political narrative is influencing perceptions, beliefs, and preferences about characters contained in the narrative. 
**Political narratives** exert their influence by depicting characters in one of the three archetipal roles: hero, villain, or victim.
These are communicative devices that focus attention, encode roles and identities, and shape norms and behavior. 

Formally, fix a topic *T* and a universe of characters 
*K = H ∪ I*, partitioned into human characters *H* (individuals or collective actors such as 
corporations, parties, states, movements) and instrument characters *I* (policies, laws, technologies).  

For any text unit (tweet, paragraph, article), let *K′ ⊆ K* be the set of characters that appear. 
A role-assignment function *r : K′ → {hero, villain, victim, neutral}* maps each appearing 
character to either a drama-triangle role or neutrality.  

We call *(T, K′, r)* a **political narrative** if and only if at least one character is cast as hero, 
villain, or victim. If all characters are neutral, the text is about the topic but does not constitute 
a political narrative in this sense.  

This definition accommodates fragments and non-sequential formulations 
(e.g., *“Corporations are villains”*) while remaining compatible with causal or temporal representations.

**How to use this guide:**
- Use the step selector above to move from **1 → 5**.
- Each step has three “cards”:
  - **Guide**: brief “How to” + reflective **Ask yourself** items ✅
  - **Example**: a concrete mini-case from an already implemented framework clarifying the step 💡
  - **Output**: what you should have before moving on ⚠️
- Jot ideas in the **Annotations** box at the end of each step, and keep comments about your progress.

**Other tabs**
- **Paper Chatbot**: ask questions about the paper and about the implementation of the Political Narratives framework.
- **Prompt Playground**: provide a short prompt and a text snippet to get a sense of what the Political Narrative framework does.
        """
    )
    # Quick start to jump into Step 1
    if st.button("Start with Step 1 →", key="intro_start_btn"):
        st.session_state["guide"]["current_step"] = 1
        st.rerun()



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

        # 1) GUIDE
        question_card(
            "Guide: Define a clear topic ✅",
            how_to=[
                "A well-defined topic is a prerequisite for fruitful narrative analysis. "
                "The clearer the topic, the more straightforward the identification of relevant characters "
                "and the exploration of the research question. Topic choice should weigh the research question, "
                "data availability, and available resources, while balancing specificity vs. generalizability. "
                "Over-narrow topics risk too few characters or narratives; over-broad topics make it difficult to "
                "restrict analysis to a manageable set. Make explicit what is in and what is out."
            ],
            ask_yourself=[
                "Does this topic surface enough distinct political narratives and public debate to analyze?",
                "Is it likely there are enough identifiable characters (actors/organizations) within those narratives?",
                "Which data sources are most informative for this topic, and do I have reliable access to them?",
                "If those sources are available, can I obtain the essential metadata (dates, outlets, geography, language) needed for analysis?",
                "Is the research question compelling and relevant to the scientific community (and/or practitioners)?",
                "Could any actors or communities be harmed by this analysis, and how will I mitigate that risk?",
                "Is the topic sufficiently specific to be analyzable, without being so narrow that it lacks variation?"
            ],
            key_prefix="s1_scope"
        )

        # 2) EXAMPLE
        example_card(
            "Focusing on policy narratives within climate change 💡",
            (
                "In *Gehring & Grigoletto (2025)* we analyze the **political economy of climate change**. "
                "From the literature we identify two dominant discussions—**scientific evidence** and **policy responses**—and, "
                "given our focus on political economy, we restrict attention to **policy narratives**, explicitly excluding "
                "debates on the scientific reality and predictability of climate change."
            ),
            key_prefix="s1_example"
        )

        # 3) OUTPUT
        output_card(
            "What you should have before Step 2 ⚠️",
            bullets=[
                "A **1–2 sentence** topic statement (domain + population + medium + lens + time/geo).",
                "**Inclusion/exclusion rules** (keywords, venues, languages).",
                "Initial **seed keywords/entities**.",
                "A brief **rationale** for timeliness and relevance."
            ],
            key_prefix="s1_output"
        )

        st.text_area("Annotations for Step 1 (optional)", key="notes_s1",
                     value=st.session_state["guide"]["notes"][1], height=120)
        st.session_state["guide"]["notes"][1] = st.session_state["notes_s1"]

    # --- STEP 2 ---
    if step == 2:
        st.subheader("Step 2 — Identify the source and extract data")
        st.caption("Choose sources (e.g., newspapers, social media, transcribed TV/radio/YouTube, surveys).")

        # 1) GUIDE
        question_card(
            "Source selection & pre-processing",
            how_to=[
                "List candidate sources and justify relevance; decide access (APIs, archives, scraping, existing corpora).",
                "Plan pre-processing: language filtering, deduplication, parsing, and metadata normalization."
            ],
            ask_yourself=[
                "Do sources match the Step-1 scope (geo/time/venue)?",
                "Is coverage balanced across time and outlets?",
                "Which formats will I parse (PDF/HTML/TXT) and how?",
                "Which metadata will I retain (date, outlet, author, section, geography, language)?"
            ],
            key_prefix="s2_sources"
        )

        # 2) EXAMPLE
        example_card(
            "National broadsheets + business dailies (2019–2024)",
            (
                "Define outlets list (e.g., *La Repubblica*, *Corriere*, *Il Sole 24 Ore*), retrieve articles containing "
                "seed terms (e.g., *“just transition”, “decarbonization”*), filter to Italian language, deduplicate wire copies, "
                "and retain metadata: **date**, **outlet**, **section**, **author** (if available)."
            ),
            key_prefix="s2_example"
        )

        # 3) OUTPUT
        output_card(
            "What you should have before Step 3",
            bullets=[
                "A **documented source list** with access method.",
                "A **pre-processing plan** (filters, de-duplication).",
                "A **sample corpus** (pilot pull) with required metadata fields.",
            ],
            key_prefix="s2_output"
        )

        st.text_area("Annotations for Step 2 (optional)", key="notes_s2",
                     value=st.session_state["guide"]["notes"][2], height=120)
        st.session_state["guide"]["notes"][2] = st.session_state["notes_s2"]

    # --- STEP 3 ---
    if step == 3:
        st.subheader("Step 3 — Identify relevant characters")
        st.caption("Map the topic into human and instrument actors with agency and claims.")

        # 1) GUIDE
        question_card(
            "Character buckets, exemplars, claims, and proxies",
            how_to=[
                "Define 4–6 buckets (government, industry, NGOs, experts, citizens, international bodies).",
                "List named exemplars per bucket; note typical claims; record missing/silenced voices.",
                "Plan detection proxies (entity lists, NER tags, acronyms, role titles)."
            ],
            ask_yourself=[
                "Which buckets matter most and why?",
                "Do I have at least 2–5 exemplars per bucket?",
                "What claims are typical for each bucket?",
                "Which actors are missing or under-quoted—and how will I surface them?",
                "Which proxies will I use to detect actors in text?"
            ],
            key_prefix="s3_chars"
        )

        # 2) EXAMPLE
        example_card(
            "Buckets and exemplars for just transition in Italy",
            (
                "**Government:** Ministry of Environment; **Industry:** ENEL, Confindustria; "
                "**Labor:** CGIL, CISL; **NGOs:** Legambiente; **Experts:** university economists; **Citizens:** local committees. "
                "Claims include **competitiveness**, **phased timelines**, **worker protection**, **environmental justice**."
            ),
            key_prefix="s3_example"
        )

        # 3) OUTPUT
        output_card(
            "What you should have before Step 4",
            bullets=[
                "A **character schema** (buckets + named exemplars).",
                "A list of **typical claims** per bucket and **omissions** to watch for.",
                "A **proxy list** for detection (entity lexicons, regex/NER patterns)."
            ],
            key_prefix="s3_output"
        )

        st.text_area("Annotations for Step 3 (optional)", key="notes_s3",
                     value=st.session_state["guide"]["notes"][3], height=120)
        st.session_state["guide"]["notes"][3] = st.session_state["notes_s3"]

    # --- STEP 4 ---
    if step == 4:
        st.subheader("Step 4 — Prepare the prompt(s)")
        st.caption("Specify the mapping from raw text to (M, R) with a simple, consistent schema.")

        # 1) GUIDE
        question_card(
            "Task, input unit, schema, and guardrails",
            how_to=[
                "Choose ONE main task (classify/extract/summarize/compare/generate) and a text unit (headline/lead/paragraph/article).",
                "Define a JSON schema (keys, allowed labels, brief rationale) and guardrails (cite spans, no external knowledge, be concise).",
                "Add 2–4 worked examples (cover easy + tricky)."
            ],
            ask_yourself=[
                "Is the task singular and clear?",
                "Is the unit appropriate for context vs. speed?",
                "Is the schema unambiguous and machine-readable?",
                "Do I include few-shots, including edge cases?"
            ],
            key_prefix="s4_prompt"
        )

        # 2) EXAMPLE
        example_card(
            "Frame classification with actor extraction (JSON)",
            (
                "```json\n"
                "{\n"
                '  "frame": "responsibility|justice|solutions",\n'
                '  "actors": ["..."],\n'
                '  "rationale": "<1-2 sentences citing spans>"\n'
                "}\n"
                "```\n"
                "_Guardrails_: Use only provided text; cite quoted spans; keep rationale ≤ 2 sentences."
            ),
            key_prefix="s4_example"
        )

        # 3) OUTPUT
        output_card(
            "What you should have before Step 5",
            bullets=[
                "A finalized **prompt spec**: task, input unit, schema, guardrails.",
                "A set of **few-shot examples** (good and near-miss)."
            ],
            key_prefix="s4_output"
        )

        st.text_area("Annotations for Step 4 (optional)", key="notes_s4",
                     value=st.session_state["guide"]["notes"][4], height=120)
        st.session_state["guide"]["notes"][4] = st.session_state["notes_s4"]

    # --- STEP 5 ---
    if step == 5:
        st.subheader("Step 5 — Obtain predictions and assemble outputs")
        st.caption("Run annotation, parse JSON, and build tidy outputs (stage flags, presence, role dummies).")

        # 1) GUIDE
        question_card(
            "Annotation runs, storage, QC, and assembly",
            how_to=[
                "Decide batch size, retries/timeouts; store JSONL/CSV with doc_id, span_id, label, rationale, annotator, timestamp.",
                "Run a small pilot; compute agreement/self-consistency; add gold items; audit outputs.",
                "Assemble tidy panel with stage-1 flags, presence m_i,k, and role dummies r_i,k."
            ],
            ask_yourself=[
                "What pilot size and QC checks will I run first?",
                "Where do parsed outputs live and how will I version them?",
                "Which metrics indicate acceptable quality, and how will I visualize/audit?"
            ],
            key_prefix="s5_outputs"
        )

        # 2) EXAMPLE
        example_card(
            "Files produced and QC pass",
            (
                "- `annotations.jsonl` (one record per unit)\n"
                "- `parsed.csv` (schema-conformant table)\n"
                "- `qc_report.md` (agreement/self-consistency + sample audits)\n"
                "- `changelog.md` (prompt/schema revisions)\n"
            ),
            key_prefix="s5_example"
        )

        # 3) OUTPUT
        output_card(
            "What you should have at the end",
            bullets=[
                "A **clean annotations file** (JSONL/CSV) matching your schema.",
                "A **QC summary** (agreement or stability checks, audits).",
                "A **tidy analysis table** (flags, presence, role dummies) ready for modeling/visualization."
            ],
            key_prefix="s5_output"
        )

        st.text_area("Annotations for Step 5 (optional)", key="notes_s5",
                     value=st.session_state["guide"]["notes"][5], height=120)
        st.session_state["guide"]["notes"][5] = st.session_state["notes_s5"]

def render_guide_tab():
    _init_guide_state()
    st.markdown("Use this walkthrough to plan your pipeline. Nothing is mandatory — mark items you’ve considered and jot notes.")

    # Options and current index (use index=..., not selection=...)
    options = ["Intro", 1, 2, 3, 4, 5]
    try:
        curr = st.session_state["guide"]["current_step"]
    except KeyError:
        curr = "Intro"
    idx = options.index(curr) if curr in options else 0

    selection = st.segmented_control(
        "Steps",
        options=options,
        index=idx,                         # <-- supported on all versions
        format_func=lambda v: "Intro" if v == "Intro" else {
            1: "1 • Topic", 2: "2 • Data", 3: "3 • Characters", 4: "4 • Prompts", 5: "5 • Outputs"
        }[v],
        key="guide_step_selector",
    )
    # Keep our own state in sync with widget output
    st.session_state["guide"]["current_step"] = selection

    # Sidebar progress
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

    # Body
    if selection == "Intro":
        render_intro()
    else:
        render_step(selection)


# ───────────────────────── Tabs ─────────────────────────
tab_guide, tab_qa = st.tabs(["Guide (5-step pipeline)", "Paper Q&A"])

# Tab 1 — GUIDE
with tab_guide:
    render_guide_tab()

# Tab 2 — Q&A (your original flow)
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
