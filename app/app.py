import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


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
OPENAI_MAX_TOKENS = 2000
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
st.set_page_config(page_title="Political Narratives Guide", layout="wide")
st.title("Political Narratives Guide")

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

# ───────────────────────── Build guide KB (Intro + Steps) ─────────────────────────
def build_guide_context() -> str:
    """Compact knowledge base from the Intro and all 5 steps to steer the LLM."""
    intro = (
        "INTRO SUMMARY\n"
        "Political narratives influence perceptions, beliefs, and preferences by assigning characters to three archetypal "
        "roles: hero, villain, or victim. Fix a topic T and a universe of characters K partitioned into human and "
        "instrument characters. A text unit (e.g., tweet, paragraph, article) constitutes a political narrative if at least "
        "one appearing character is cast as hero, villain, or victim; if all are neutral, the text is about the topic but does "
        "not form a political narrative in this strict sense. The pipeline has 5 steps: Topic → Data → Characters → Prompts → Outputs.\n"
    )

    # ---------- STEP 1 ----------
    step1_how = (
        "- Start from a well-defined research question and articulate the topic narrowly enough to be analyzable, yet broad enough to yield variation.\n"
        "- Make explicit what is IN and what is OUT of scope to prevent drift in downstream steps (characters, prompts, audits).\n"
        "- Balance specificity (clear boundaries, better precision) vs. generality (transferable insights, broader coverage).\n"
        "- Check feasibility early: ensure there will be enough content, observable variation, and a manageable set of characters.\n"
        "- Confirm basic metadata needs can be met later (dates, outlets, geography, language) because these enable grouping, fixed effects, and robustness checks.\n"
    )
    step1_ask = (
        "- Does this topic surface enough distinct narratives and public debate to analyze?\n"
        "- Are there likely to be identifiable characters (actors, organizations, policies) within those narratives?\n"
        "- Which sources best capture discourse on this topic, and will I have reliable access to them?\n"
        "- Can I get essential metadata (dates, outlets, geography, language) for downstream analysis?\n"
        "- Is the question interesting to researchers/practitioners, and are potential harms identified with mitigation plans?\n"
        "- Is the scope precise enough to be analyzable without being so narrow that it yields too little variation?\n"
    )

    # ---------- STEP 2 ----------
    step2_how = (
        "- Choose data sources where the debate actually unfolds (e.g., newspapers, social media, transcribed TV/radio/YouTube, surveys).\n"
        "- Evaluate coverage (breadth, time window), accessibility (APIs, licenses), and quality (OCR errors, platform bias, sampling limits).\n"
        "- Decide the extraction approach suited to the source: keyword queries, metadata filters, API pulls, scraping, or manual curation.\n"
        "- Preserve rich metadata (dates, outlets, authors, geography, language) to enable context-aware analysis and identification strategies.\n"
        "- Aim for snippet lengths that preserve interpretability for the LLM without overshooting context limits.\n"
    )
    step2_ask = (
        "- Do these sources capture the main venues of the political debate for my topic?\n"
        "- Are sources diverse enough to avoid ideological/outlet/demographic bias?\n"
        "- Is the extraction method precise (relevant results, manageable false positives/negatives)?\n"
        "- Do I have legal and technical access (archives, APIs, permission to scrape)?\n"
        "- Is the time window aligned with the question (pre/post events, policy cycles)?\n"
        "- Will I retain essential metadata for downstream grouping and robustness?\n"
    )

    # ---------- STEP 3 ----------
    step3_how = (
        "- Identify a focused set of characters that speak directly to your research question—both human (individuals/groups/organizations/states) "
        "and instrument (policies, tools, institutions, technologies).\n"
        "- Balance scope vs. feasibility: too many characters reduce precision and inflate costs; a concise list improves reliability and interpretability.\n"
        "- Build the list via literature review, exploratory tools (topic models, entity recognition, RELATIO), and domain reading; document inclusion/exclusion decisions.\n"
        "- Prefer characters that can plausibly appear in multiple roles (hero, villain, victim) to capture narrative variation.\n"
    )
    step3_ask = (
        "- Which entities matter most for explaining outcomes in my question?\n"
        "- Should I treat them as human actors or instrument actors, and why?\n"
        "- What is the analytic scope (national, regional, global) and which entities are out of scope?\n"
        "- Which characters recur in prior literature or appear prominently in exploratory tools?\n"
        "- How many characters can I reliably track while keeping labeling interpretable and statistically useful?\n"
        "- Are character definitions unambiguous enough for LLM labeling across contexts?\n"
    )

    step3_example = (
        "STEP 3 EXAMPLE\n"
        "Human: DEVELOPING ECONOMIES | US DEMOCRATS | US REPUBLICANS | CORPORATIONS | US PEOPLE\n"
        "Instrument: EMISSION PRICING | REGULATIONS | FOSSIL INDUSTRY | GREEN TECH | NUCLEAR TECH\n"
        "Rationale: Ten characters balance analytic richness with annotation reliability and cost. Clear descriptions provided in Step 4 prompts "
        "help the model consistently detect mentions and roles.\n"
    )

    # ---------- STEP 4 ----------
    step4_how = (
        "- Choose ONE main task and ONE input unit (e.g., classify a tweet, extract from a paragraph) to avoid prompt sprawl.\n"
        "- Define a minimal **JSON schema** with stable keys, allowed labels, and short rationales, so outputs are machine-readable and easy to parse.\n"
        "- Add **2–4 worked examples**, including near-miss edge cases, to anchor the model’s decisions and reduce ambiguity.\n"
        "- Specify guardrails (cite spans if possible, no external knowledge, be concise, follow schema strictly).\n"
        "- Co-design prompts with the same model you will use for annotation to align capabilities and output formats.\n"
        "- This workflow supports two tasks: (a) topic relevance classification; (b) character detection + role assignment.\n"
    )
    step4_ask = (
        "- Is the task singular and clearly stated (no multi-task mixing)?\n"
        "- Is the input unit appropriate for context vs. latency/throughput trade-offs?\n"
        "- Is the schema unambiguous, stable, and easy to parse downstream?\n"
        "- Do few-shots cover both straightforward and tricky cases (near-misses)?\n"
        "- Are character descriptions precise enough for consistent detection and role assignment?\n"
        "- Have I added guardrails (no external knowledge, cite spans when possible, be concise)?\n"
    )

    # ---------- STEP 5 ----------
    step5_how = (
        "- Prepare inputs: (1) a dataset with observation ID and text snippet fields; (2) finalized prompts for the two tasks (relevance; character+role).\n"
        "- Verify local folder structure matches the repository layout so paths for data, prompts, and outputs resolve correctly.\n"
        "- Decide batch size, retries, and logging. Store outputs in JSONL/CSV with document ID, labels, rationales (if any), and timestamps.\n"
        "- Pilot first: check agreement/self-consistency, maintain a small set of gold items, and audit failure modes before scaling.\n"
        "- Assemble the tidy table comprising: stage flags (relevance), character presence indicators, and role dummies.\n"
        "- Version outputs and keep an audit trail so results are reproducible (model, parameters, prompt versions, time of run).\n"
    )
    step5_ask = (
        "- Is my dataset clean (IDs present, text present, no unexpected missing values)?\n"
        "- Are the two prompts finalized, specific, and aligned with the selected model?\n"
        "- Does my local folder structure mirror the repository so all scripts can run end-to-end?\n"
        "- What QC metrics will I monitor (agreement, self-consistency, error audits), and where will I store them?\n"
        "- How will I version outputs (filenames, folders, or DVC/Git) for reproducibility?\n"
    )

    return (
        intro
        + "\nSTEP 1 – TOPIC\n" + step1_how + step1_ask
        + "\nSTEP 2 – DATA\n" + step2_how + step2_ask
        + "\nSTEP 3 – CHARACTERS\n" + step3_how + step3_ask
        + "\n" + step3_example
        + "\nSTEP 4 – PROMPTS\n" + step4_how + step4_ask
        + "\nSTEP 5 – OUTPUTS\n" + step5_how + step5_ask
    )


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

GUIDE_CONTEXT = build_guide_context()

GUIDE_AWARE_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise, independent research assistant for the Political Narratives project.\n\n"
    "HOW TO ANSWER:\n"
    "1) Start with a direct answer to the user's question in 2–4 sentences.\n"
    "2) Only reference Guide steps when explicitly asked about a step (e.g., 'Step 3', 'characters') or when it clearly adds value.\n"
    "   If so, provide a focused deep dive into the single most relevant step (do NOT enumerate all steps).\n"
    "   Use the exact step label (e.g., 'Step 3 – Characters') and summarize the key guidance.\n"
    "3) If the user asks about core definitions or concepts (e.g., 'What is a political narrative?'), use the Intro to explain clearly.\n"
    "4) Add practical tips/examples from your own judgment when useful, but keep them aligned with the Guide.\n"
    "5) Support factual claims with short evidence from the paper excerpts when possible; if details are missing, say so briefly.\n\n"
    "GUIDE (Intro + Steps 1–5):\n"
    f"{GUIDE_CONTEXT}\n\n"
    "PAPER EXCERPTS (for evidence/details):\n"
    "{context}\n\n"
    "USER QUESTION:\n"
    "{question}\n\n"
    "RESPONSE FORMAT:\n"
    "- Direct answer very detailed also from your own knowledge, not only from the steps and guide.\n"
    "- If relevant: Deep dive on the single most relevant Step N (or Intro concept), with 3–6 bullets.\n"
    "- Optional next actions: 3–5 checklist items.\n"
    "Style: conversational, concise, and confident like ChatGPT; avoid listing all steps unless asked."
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
    st.subheader("Political Narratives guide")
    st.markdown(
        """
The purpose of a political narrative is influencing perceptions, beliefs, and preferences about characters contained in the narrative.  **Political narratives** exert their influence by depicting characters in one of the three archetypal roles—**hero**, **villain**, or **victim**.  They are communicative devices that focus attention, encode roles and identities, and shape norms and behavior.

Formally, fix a topic *T* and a universe of characters *K = H ∪ I*, partitioned into human characters *H* (individuals or collective actors such as corporations, parties, states, movements) and instrument characters *I* (policies, laws, technologies). For any text unit (tweet, paragraph, article), let *K′ ⊆ K* be the set of characters that appear.  
A role-assignment function *r : K′ → {hero, villain, victim, neutral}* maps each appearing character to either a drama-triangle role or neutrality. We call *(T, K′, r)* a **political narrative** if and only if at least one character is cast as hero, villain, or victim; if all characters are neutral, the text is about the topic but does not constitute a political narrative in this sense.  

**How to use this guide**
- Use the step selector above to move from **1 → 5**.
- Each step includes three cards:
  - **Guide ✅** — brief “How to” plus reflective **Ask yourself** items.
  - **Example 💡** — a concrete mini-case clarifying the step.
  - **Output ⚠️** — what you should have before moving on.
- Jot ideas in the **Annotations** box at the end of each step and keep notes on your progress.

**Other tabs**
- **Paper Q&A** — ask questions about the paper and the Political Narratives framework.
- **Prompt Playground** — try a short prompt and text snippet to see the framework in action.
        """
    )

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
            "Guide: define a clear topic ✅",
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
                "A precise topic for the analysis of Political Narratives",
                "Availability of a source for the data extraction"
            ],
            key_prefix="s1_output"
        )

        st.text_area("Personal comments for Step 1", key="notes_s1",
                     value=st.session_state["guide"]["notes"][1], height=120)
        st.session_state["guide"]["notes"][1] = st.session_state["notes_s1"]

    # --- STEP 2 ---
    if step == 2:
        st.subheader("Step 2 — Identify the source and extract data")
        st.caption("Choose sources (e.g., newspapers, social media, transcribed TV/radio/YouTube, surveys).")

        # 1) GUIDE
        question_card(
            "Guide: source selection & data extraction ✅",
            how_to=[
                "After selecting the topic, the next step is gathering data. Common sources include digitized newspapers, social media, transcribed TV/radio/YouTube content, and open-ended survey responses.",
                "When selecting the data source, prioritize the media channels where narratives about your chosen topic are most prominent.",
                "Evaluate trade-offs between coverage, accessibility, and quality (e.g., digitization errors, platform bias, sampling limits).",
                "For data extraction, the chosen source will determine which methodologies can be applied—such as keyword-based queries, scraping, API pulls, or manual collection.",
                "Consider the level of metadata you can preserve (dates, outlets, authors, geography, language) since these details will later be the basis of your analysis."
            ],
            ask_yourself=[
                "Do the chosen sources capture the main media where the political debate unfolds?",
                "Are they sufficiently diverse to avoid bias toward one outlet, ideology, or demographic?",
                "Is the extraction method able to produce snippets that are neither too short to lose context nor too long to become too complicated for the analysis?",
                "Do I have legal and technical access to these data (e.g., archives, APIs, scraping permission)?",
                "What extraction method is most reliable for my source—keyword queries, metadata filters, or transcript parsing?",
                "How will I ensure that the collected snippets are relevant to the topic and not dominated by noise?",
                "Is the time window covered by the source appropriate for the research question?",
                "Can I obtain essential metadata (dates, outlets, geography, language) for contextual analysis?"
            ],
            key_prefix="s2_sources"
        )

        # 2) EXAMPLE
        example_card(
            "Extracting data using Twitter Historical APIv2 💡",
            (
                "In *Gehring & Grigoletto (2025)* our focus is on narratives about climate change policies in the United States, collected from the social media platform Twitter. "
                "We specifically choose the U.S. due to the significant role Twitter plays in shaping and disseminating political narratives there. "
                "The data collection process involves querying the **Twitter historical APIv2** with a set of **keywords** adapted from *Oehl, Schaffer, and Bernauer (2017)*"
            ),
            key_prefix="s1_example"
        )

        # 3) OUTPUT
        output_card(
            "What you should have before Step 3 ⚠️",
            bullets=[
                "A dataset with the extracted text snippets and other relevant information for your analysis.",
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
        # 1) GUIDE
        question_card(
            "Guide: character selection ✅",
            how_to=[
                "Identify relevant characters for the topic (human and instrument) — this is the core of Step 3.",
                "Anchor selection in your research question and analytical focus; choose characters that speak to your hypotheses.",
                "Balance scope with feasibility: too many characters can reduce precision and raise compute costs; a focused set improves reliability and interpretability.",
                "Build the character list via literature review, exploratory tools (topic modeling, entity recognition/RELATIO), and domain reading; document your choices."
            ],
            ask_yourself=[
                # Conceptual framing
                "Which entities matter most for the topic at hand?",
                "Are these best understood as human actors (individuals, groups, organizations, states) or instrumental actors (policies, tools, institutions)?",
                "What is the scope of my analysis (national, regional, global)? Which entities fall outside my scope?",

                # Selection criteria
                "Which characters recur most often in prior literature or theory on this topic?",
                "Do exploratory tools (topic models, word clouds, entity recognition, RELATIO outputs) highlight additional frequently mentioned entities?",
                "How many characters can I realistically track while keeping the coding interpretable and statistically useful?",

                # Practical implementation
                "Is the chosen character list feasible for LLM coding (not too long, not too ambiguous)?",
                "Can each character plausibly appear in different roles (hero, villain, victim), or are some inherently neutral/instrumental?",
            ],
            key_prefix="s3_chars"
        )

        # 2) EXAMPLE
        example_card(
            "Relevant characters for climate change political discourse 💡",
            (
                "Guided by the relevant literature, exploratory tools, and intensive domain reading, "
                "we pre-specify ten characters: five human characters (made of institutions and groups of individuals) "
                "and five instrument characters (policy tools and instruments).\n\n"
                "**Human Characters:** DEVELOPING ECONOMIES | US DEMOCRATS | US REPUBLICANS | CORPORATIONS | US PEOPLE\n\n"
                "**Instrument Characters:** EMISSION PRICING | REGULATIONS | FOSSIL INDUSTRY | GREEN TECH | NUCLEAR TECH\n\n"
                "We carefully decided to have ten character to balance the complexity of the analysis with "
                "good practices to avoid overloading the LLM in the annotation process. These characters are easily recognizable "
                "by the LLM thanks to precise descriptions in the prompt design step (Step 4)."
            ),
            key_prefix="s3_example"
        )

        # 3) OUTPUT
        output_card(
            "Output - What you should have before Step 4 ⚠️",
            bullets=[
                "A contained list of relevant characters, both in human and instrument form",
                "A set of descriptions for these characters is needed in the next step, so it's good practice to annotate these during this stage (useful, but not necessary now)"
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
            "Guide: prepare the prompt(s) ✅",
            how_to=[
                "Choose ONE main task (classify / extract / summarize / compare / generate) and one input unit (headline / paragraph / article / tweet).",
                "Define a **JSON schema** (keys, allowed labels, brief rationale) and **guardrails** (e.g., cite spans, no external knowledge, be concise).",
                "Add **2–4 worked examples** (cover easy + tricky edge cases).",
                "Co-design prompts with the same model you will use for annotation (e.g., GPT-4o-mini via OpenAI) to align capabilities and outputs.",
                "This package supports two tasks: (a) topic relevance classification and (b) character detection + role assignment."
            ],
            ask_yourself=[
                "Is the task singular and clear (avoid mixing multiple tasks in one prompt)?",
                "Is the chosen input unit appropriate for context vs. speed constraints?",
                "Is the schema unambiguous, machine-readable, and easy to parse?",
                "Do my few-shot examples include near-miss/edge cases?",
                "Do I provide a description of the selected characters?",
                "Have I specified guardrails (no external knowledge, cite spans, be concise)?"
            ],
            key_prefix="s4_prompt"
        )

        # 2) EXAMPLE
        example_card(
            "Prompt design for the political narratives of climate change 💡",
            (
                "##### (a) Relevance classifier (Stage 1)\n"
                "Decide if a three-sentence newspaper excerpt is relevant to **US climate change discourse**.\n"
                "- 0 = irrelevant (no meaningful climate discussion)\n"
                "- 1 = assert (asserts existence of climate change)\n"
                "- 2 = deny (denies or mocks climate change)\n"
                "- 3 = relevant (substantive discussion of climate change policy)\n\n"
                "##### (b) Character detection & role assignment (Stage 2)\n"
                "Identify pre-specified characters and assign contextual roles:\n"
                "- Villain (1) | Hero (2) | Victim (3) | No role (4)\n\n"
                "Characters: Developing Economies, US Democrats, US Republicans, Corporations, US People, "
                "Emission Pricing Tools, Regulation Policies, Fossil Fuels, Green Technologies, Nuclear Energy.\n"
                "Output JSON contains keys `a`–`j` (one per character)."
            ),
            key_prefix="s4_example"
        )

        # 3) OUTPUT
        output_card(
            "What you should have before Step 5 ⚠️",
            bullets=[
                "A finalized **prompt** for both tasks (relevance + character/role), including schema and guardrails.",
                "A set of **few-shot examples** (both positive and near-miss).",
                "Consistent key names to enable parsing into (M, R) downstream."
            ],
            key_prefix="s4_output"
        )

        # ——— PROMPT PASTE AREAS (stores in session_state) ———
        st.markdown("### Paste your JSON prompts")
        st.caption("Provide machine-readable JSON specs for Stage 1 (relevance classification) and Stage 2 (character/role assignment). You can find the original prompts for *Gehring and Grigoletto (2025) in the GitHub repository.")

        # Load defaults from uploaded JSON system messages
        default_relevance_json = """{
  "SYSTEM_MESSAGE": "You are an average US citizen. The user will provide the content of a three-sentence newspaper article excerpt published by a US newspaper between 2010 and 2021. 
Your task is to analyze the excerpt within the context of US political discourse, particularly in relation to climate change. Respond in JSON format.

1. Relevance Check: Analyze the excerpt in the context of US climate change discussion and determine its relevance. Provide one of the following values:
   - 0 (irrelevant): If the excerpt does not discuss climate change in a meaningful way. For example, if it only includes a hashtag (like #climatechange) or a passing reference but does not engage in any discussion about climate change or related policies, it should be considered irrelevant.
   - 1 (assert): If the excerpt asserts the existence of climate change but does not engage with specific policies or actions related to it. This includes excerpts that acknowledge climate change as an issue without going deeper into details.
   - 2 (deny): If the excerpt denies the existence or severity of man-made climate change, referring to it as a hoax, scam, or fraud, or using sarcasm or language that undermines the reality of climate change.
   - 3 (relevant): If the excerpt discusses climate change or related policies in a substantive way. This includes any excerpt that debates, critiques, or supports policies or actions related to climate change, as well as conversations on how to combat or adapt to climate change.

Respond in JSON format, returning the value in the key \\"r\\"."
}"""

        default_roles_json = """{
  "SYSTEM_MESSAGE": "You are an average US citizen. The user will provide a three-sentence US newspaper excerpt (2010–2021). 
Analyze it in the context of US political discourse on climate change and respond in JSON format.

1. Character Analysis: For each mentioned character, assign a role:
   - 1 = Villain: contributes to problems, opposes positive change, engages in harmful actions.
   - 2 = Hero: leads efforts to combat climate change, promotes environmental policies, acts commendably.
   - 3 = Victim: suffers unfairly, is attacked, or endures consequences of climate change or others’ actions.
   - 4 = No role: mentioned but not clearly cast as Villain/Hero/Victim, or context is neutral/ambiguous.

2. Characters to evaluate (keys a–j):
   a: Developing Economies — emerging and poorer nations (incl. BRICS). Mentions of their governments, representatives, or citizens in the context of climate negotiations or responsibilities.
   b: US Democrats — politicians or public figures tied to the Democratic Party (e.g., Biden, Obama, Pelosi). References to Democratic climate policies, proposals, or positions.
   c: US Republicans — politicians or public figures tied to the Republican Party (e.g., Trump, McConnell, Cruz). Mentions of opposition or support to climate-related policies from the GOP side.
   d: Corporations and Industry — private-sector actors including large companies, SMEs, banks, CEOs, and industry lobbies. Includes mentions of energy, tech, finance, or manufacturing in the climate context.
   e: US People — the collective public, workers, voters, youth, and grassroots movements (e.g., Sunrise Movement, Extinction Rebellion). Includes references to citizens as beneficiaries, victims, or agents of climate action.
   f: Emission Pricing Tools — market-based policies like carbon taxes, cap-and-trade, carbon markets, or pollution credits. References to pricing carbon as a solution or a burden.
   g: Regulation Policies — government bans or regulations (e.g., banning fracking, fossil phaseouts, plastics bans, degrowth/anti-capitalist proposals). Mentions of regulatory tools to address climate change.
   h: Fossil Fuels — oil, gas, coal, and related infrastructure (power plants, pipelines). Mentions in the context of pollution, energy needs, or phaseout debates.
   i: Green Technologies — renewable and low-carbon solutions like solar, wind, EVs, hydrogen, batteries, geothermal. Mentions of their potential, adoption, or challenges.
   j: Nuclear Energy — nuclear power, fission, or fusion technologies. Mentions of nuclear plants or research in the climate policy debate.

3. Final Output: Respond with a JSON object containing keys a–j. Each value must be 0 (not mentioned) or 1–4 (role as defined above)."
}"""


        # Initialize session storage
        st.session_state.setdefault("s4_relevance_json", default_relevance_json)
        st.session_state.setdefault("s4_roles_json", default_roles_json)

        st.text_area(
            "🧩 (a) Stage 1 — Relevance prompt JSON",
            key="s4_relevance_json",
            height=320
        )

        st.text_area(
            "🧩 (b) Stage 2 — Character & role prompt JSON",
            key="s4_roles_json",
            height=500
        )

        # Optional free notes area (consistent with other steps)
        st.text_area(
            "Annotations for Step 4 (optional)",
            key="notes_s4",
            value=st.session_state["guide"]["notes"][4],
            height=120
        )
        st.session_state["guide"]["notes"][4] = st.session_state["notes_s4"]

    # --- STEP 5 ---
    if step == 5:
        st.subheader("Step 5 — Obtain predictions and assemble outputs")
        st.caption("Run annotation, parse JSON, and build tidy outputs (stage flags, presence, role dummies).")

        # 1) GUIDE
        question_card(
            "Guide - GPT Annotation with provided code ✅",
            how_to=[
            "After the design of the prompts, it's time to run the code to query OpenAI GPT 4o-mini model to perform the task of interest. For this phase it's crucial to have the following two as inputs:",
            "1) A dataset with observation id, text snippets. Other variables will be discarded before running the main functions.",
            "2) The two prompts, one for each task for GPT model.",
            "Before running, check the folder structure described in the Github repository and replicate it on your machine."
            ],
            ask_yourself=[
                "Is my dataset ready for the annotation (id, text, missing values, etc)?",
                "Are my prompts correctly specified and specific enough?",
                "Is my folder structure organised as the one suggested by the authors?"
            ],
            key_prefix="s5_outputs"
        )

        # 2) EXAMPLE
        example_card(
            "Annotation output and relevant tweets 💡",
            (
                "In our paper the annotation produces a tidy panel with:\n"
                "- Stage-1 flags (relevance of the tweet)\n"
                "- Character presence (which characters appear)\n"
                "- Role indicators (whether a character is assigned a role)\n\n"
                "Tweets are defined as relevant if they refer to our topic and contain at least one of the "
                "pre-specified characters (neutral or with a role). This yields **309,744 relevant tweets**.\n\n"
                "Within relevant tweets, a political narrative exists if at least one role assignment is present. "
                "All other relevant tweets contain characters but with no roles, i.e., in a neutral way.\n\n"
            ),
            key_prefix="s4_annotation_example"
        )


        # 3) OUTPUT
        output_card(
            "What you should have at the end 🎆",
            bullets=[
                "A **clean annotations file** in .csv format matching your schema. 🎆",
            ],
            key_prefix="s5_output"
        )

        st.text_area("Annotations for Step 5 (optional)", key="notes_s5",
                     value=st.session_state["guide"]["notes"][5], height=120)
        st.session_state["guide"]["notes"][5] = st.session_state["notes_s5"]

def render_guide_tab():
    _init_guide_state()
    st.markdown("Use this guide to understand what the concept of **Political Narrative** is and to organize the pipeline for your research. Nothing is mandatory—mark items you’ve considered and jot notes.")

    # Step selector now includes an Intro page (default selected)
    step = st.segmented_control(
        "Steps",
        options=["Intro", 1, 2, 3, 4, 5],
        format_func=lambda v: "Intro" if v == "Intro" else {1:"1 • Topic", 2:"2 • Data", 3:"3 • Characters", 4:"4 • Prompts", 5:"5 • Outputs"}[v],
        key="guide_step_selector"
    )

    # Sidebar progress: a step turns green only when ALL its checkboxes are marked
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

    # Body: render Intro or a numbered step
    if step == "Intro":
        render_intro()
    else:
        render_step(step)

# ───────────────────────── Tabs ─────────────────────────
tab_guide, tab_qa = st.tabs(["Guide (5-step pipeline)", "Paper Q&A"])

# Tab 1 — GUIDE
with tab_guide:
    render_guide_tab()

# Tab 2 — Q&A (guide-aware flow)
with tab_qa:
    st.markdown(
    "Use this Q&A to ask focused questions about the steps to perform the **Political Narratives** analysis. "
    "Be precise and reference to the steps or the paper itself."
    )
    if vs is None:
        st.info("No documents indexed. Add your paper at data/paper.pdf and rerun.")
    else:
        llm = get_llm(provider, user_key, DEFAULT_LOCAL_MODEL)

        if provider.startswith("OpenAI") and not user_key:
            st.stop()

        # Always guide-aware, no toggle
        st.markdown(
            "Answers are structured using the Intro + Steps 1–5 from the Guide tab, "
            "and supported by excerpts from the paper of reference."
        )

        retriever = vs.as_retriever(
            search_kwargs={"k": 12, "fetch_k": 24},
            search_type="mmr"
        )
        # Always use the guide-aware prompt
        qa_prompt = GUIDE_AWARE_PROMPT

       doc_chain = create_stuff_documents_chain(llm, GUIDE_AWARE_PROMPT)
qa = create_retrieval_chain(retriever, doc_chain)

# Input key must be "input"
out = qa.invoke({"input": q})

# The answer lives under "answer"
st.write(out["answer"])


