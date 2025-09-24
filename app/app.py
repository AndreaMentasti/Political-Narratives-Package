import os, glob
from datetime import date
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

# ---------------- Constants ----------------
DEFAULT_LOCAL_MODEL = "llama3.2:3b"   # try "qwen2.5:3b" if you want faster local replies
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Speed-friendly defaults
FIXED_TEMPERATURE = 0.1        # stability (not speed)
OLLAMA_KWARGS = {
    "num_predict": 256,        # hard cap on output tokens (biggest speed win)
    "num_ctx": 2048,           # context window; keep modest
    "top_k": 30,
    "top_p": 0.9,
}
OPENAI_MAX_TOKENS = 500        # keep OpenAI outputs short too

# ---- Safe ALLOW_LOCAL detection (no secrets.toml -> no crash) ----
def _read_allow_local_from_secrets() -> bool:
    try:
        return bool(st.secrets.get("ALLOW_LOCAL", False))
    except Exception:
        return False

_env_flag = os.environ.get("ALLOW_LOCAL", "").strip()
_secret_flag = _read_allow_local_from_secrets()
# Default to True locally (when nothing set) so you can use Ollama during dev
ALLOW_LOCAL = True if (_env_flag == "" and _secret_flag is False) else bool(_env_flag or _secret_flag)

# ---------------- UI Header ----------------
st.set_page_config(page_title="Political Narratives — Guide + Q&A + Playground", layout="wide")
st.title("Political Narratives — Guide + Local Q&A (RAG) + Prompt Playground")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Model settings")

    # Show only OpenAI online (default). Locally, show both providers.
    provider_options = ["OpenAI (bring your own key)"] if not ALLOW_LOCAL else ["Local (Ollama)", "OpenAI (bring your own key)"]
    default_index = 0
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

# --------------- Warmup (optional) ---------------
@st.cache_resource(show_spinner=False)
def warmup_local_model(model_name: str):
    try:
        _ = ChatOllama(model=model_name, temperature=0.0).invoke("hi")
    except Exception:
        pass

if provider.startswith("Local"):
    warmup_local_model(local_model)

# ---------------- Helpers ----------------
ASSISTANT_SYSTEM = (
    "You are a helpful research assistant and project coach for applying the Political Narratives framework, "
    "especially Drama Triangle character recognition, to political narratives. Prefer using retrieved context from "
    "the user's paper/repo when it is relevant. If the context is missing, sparse, or not directly relevant, answer "
    "from general knowledge and common sense, and be a guide to the user. Be concise and concrete: "
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
    Routing heuristic:
      1) Try FAQ.
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

# ======================= NEW: GUIDE TAB HELPERS =======================
def _init_guide_state():
    st.session_state.setdefault("guide_state", {"step": 1, "notes": {}, "reviewed": set()})

def question_card(title, how_to=None, examples=None, pitfalls=None, note_key=None, reviewed_key=None):
    """A simple boxed section to teach + (optional) jot notes. No validation."""
    with st.container(border=True):
        c1, c2 = st.columns([0.92, 0.08])
        with c1:
            st.markdown(f"**{title}**")
        with c2:
            if reviewed_key:
                checked = st.checkbox("Reviewed", value=(reviewed_key in st.session_state["guide_state"]["reviewed"]))
                if checked:
                    st.session_state["guide_state"]["reviewed"].add(reviewed_key)
                else:
                    st.session_state["guide_state"]["reviewed"].discard(reviewed_key)
        if how_to:
            st.markdown("**How to approach**")
            st.markdown("\n".join([f"- {p}" for p in how_to]))
        if examples:
            st.markdown("**Examples**")
            st.markdown("\n".join([f"- _{e}_" for e in examples]))
        if pitfalls:
            st.markdown("**Pitfalls**")
            st.markdown("\n".join([f"- {p}" for p in pitfalls]))
        if note_key:
            default_note = st.session_state["guide_state"]["notes"].get(note_key, "")
            note = st.text_area("Jot a note (optional)", value=default_note, height=70, key=f"note_{note_key}")
            st.session_state["guide_state"]["notes"][note_key] = note

def render_step1():
    st.subheader("Step 1 — Select a Topic")
    st.caption("Goal: sharpen scope, lens, and boundaries.")
    question_card(
        "Topic statement (1–2 sentences)",
        how_to=[
            "State domain + population + medium + lens; be specific.",
            "Signal scope so later steps can build corpus & codes."
        ],
        examples=[
            "Italian national newspapers framing *just transition* (2019–2024) via responsibility vs justice.",
            "Local TV news in Southern Spain portraying heatwaves as individual vs systemic issues (2016–2025)."
        ],
        pitfalls=["Avoid vague nouns like 'the media' without scope."],
        note_key="s1_topic", reviewed_key="s1_topic"
    )
    question_card(
        "Narrative lens (what you’ll foreground)",
        how_to=["Pick 1–3: responsibility, justice, risk, solutions, adaptation, mitigation, health, economics."],
        examples=["Responsibility & justice for just transition", "Risk & health for heatwaves"],
        note_key="s1_lens", reviewed_key="s1_lens"
    )
    question_card(
        "Scope (where, when, which mediums)",
        how_to=["Specify geography, time window, mediums (news/policy/social/NGO/PR/academic)."],
        examples=["Italy; 2019–2024; national broadsheets & business dailies."],
        pitfalls=["Open-ended dates; mixing too many mediums at once."],
        note_key="s1_scope", reviewed_key="s1_scope"
    )
    question_card(
        "Inclusion / exclusion rules",
        how_to=["Write 1–2 bullets each: language, headline/lead keywords, outlet list; exclude duplicates/wires/op-eds if needed."],
        examples=["Include: 'just transition' in headline; Exclude: syndicated wire copies."],
        note_key="s1_rules", reviewed_key="s1_rules"
    )
    question_card(
        "Seeds (keywords and entities)",
        how_to=["List 5–15 terms + 5–10 named orgs/people; include local spellings."],
        examples=["KW: just transition, decarbonization; ENT: ENEL, CGIL, Confindustria"],
        pitfalls=["Only one synonym; forgetting local variants."],
        note_key="s1_seeds", reviewed_key="s1_seeds"
    )
    question_card(
        "Why this topic now?",
        how_to=["Link to decisions, policy windows, and stakeholders who care."],
        examples=["EU Fit for 55 roll-out shifted responsibility narratives."],
        note_key="s1_why", reviewed_key="s1_why"
    )
    st.info("**Carry forward:** topic statement, lens, scope (time/geo/mediums), include/exclude rules, seeds, rationale.")

def render_step2():
    st.subheader("Step 2 — Select a Research Question")
    st.caption("Goal: from interest → answerable question.")
    question_card(
        "Primary research question",
        how_to=["Choose a verb that signals method: describe / explain / compare / influence."],
        examples=[
            "Describe: What frames dominate coverage of X?",
            "Explain: What predicts shifts toward frame Y?",
            "Compare: How do outlet A vs B differ on Z?",
            "Influence: Which counter-narratives reduce misinformation about W?"
        ],
        note_key="s2_rq", reviewed_key="s2_rq"
    )
    question_card("Sub-questions (optional)",
        how_to=["Break the RQ into 2–3 tractable angles (actors, claims, tone)."],
        examples=["Which actors are most quoted when responsibility is assigned?"],
        note_key="s2_subq", reviewed_key="s2_subq"
    )
    question_card("Analytic lens or theory",
        how_to=["Name it (framing, narrative policy framework, discourse analysis) to guide coding & outputs."],
        note_key="s2_theory", reviewed_key="s2_theory"
    )
    question_card("Falsification (what would disconfirm your hunch)",
        how_to=["Write 1–2 concrete counter-signals you would accept as disconfirming evidence."],
        examples=["Responsibility frames do **not** increase after policy announcements."],
        note_key="s2_falsify", reviewed_key="s2_falsify"
    )
    question_card("Decision use (who uses this and for what)",
        how_to=["Tie the answer to a decision-maker and a decision."],
        examples=["NGO comms team decides which counter-frames to emphasize."],
        note_key="s2_decision", reviewed_key="s2_decision"
    )
    st.info("**Carry forward:** RQ, intent, sub-questions, lens/theory, falsification signals, decision user.")

def render_step3():
    st.subheader("Step 3 — Select Characters (actors & voices)")
    st.caption("Goal: map agents & their claims.")
    question_card("Character buckets",
        how_to=["Aim for 4–6 buckets (government, industry, NGOs, experts, citizens)."],
        examples=["Government • Industry • Civil society • Experts • Citizens"],
        note_key="s3_buckets", reviewed_key="s3_buckets"
    )
    question_card("Named exemplars",
        how_to=["List 2–5 per bucket (orgs, agencies, firms, unions)."],
        examples=["Ministry of Environment, ENEL, CGIL, Legambiente"],
        note_key="s3_exemplars", reviewed_key="s3_exemplars"
    )
    question_card("Typical claims per bucket",
        how_to=["1–2 example claims each (responsibility, costs, feasibility)."],
        examples=["Industry → competitiveness & phased timelines"],
        note_key="s3_claims", reviewed_key="s3_claims"
    )
    question_card("Who is missing or silenced?",
        how_to=["Actors affected but rarely quoted; plan to surface them."],
        examples=["Informal workers; transition towns’ local communities"],
        note_key="s3_omissions", reviewed_key="s3_omissions"
    )
    question_card("Proxies to detect characters in text",
        how_to=["Named lists, NER tags, regex for acronyms, role titles."],
        examples=["ORG list for utilities; PER for ministers; 'Ministry of …' patterns"],
        note_key="s3_proxies", reviewed_key="s3_proxies"
    )
    st.info("**Carry forward:** buckets, exemplars, claims, omissions, detection proxies.")

def render_step4():
    st.subheader("Step 4 — Prompt Design")
    st.caption("Goal: translate plan into robust prompts.")
    question_card("Core task",
        how_to=["Pick one: classify / extract / summarize / compare / generate."],
        examples=["Classify an article into frames: {responsibility, justice, solutions}"],
        note_key="s4_task", reviewed_key="s4_task"
    )
    question_card("Inputs (text unit)",
        how_to=["Choose: headline, lead, paragraph, full article (trade speed vs context)."],
        note_key="s4_inputs", reviewed_key="s4_inputs"
    )
    question_card("Output format (schema & constraints)",
        how_to=["Specify JSON keys, allowed labels, and a short rationale."],
        examples=['{"frame":"responsibility|justice|solutions","rationale":"<1-2 sentences>","actors":["..."]}'],
        note_key="s4_schema", reviewed_key="s4_schema"
    )
    question_card("Style constraints & guardrails",
        how_to=["No external knowledge; cite spans; be concise; use only given text."],
        note_key="s4_guardrails", reviewed_key="s4_guardrails"
    )
    question_card("Worked examples (few-shot)",
        how_to=["2–4 examples: easy + tricky + counter-examples."],
        note_key="s4_fewshot", reviewed_key="s4_fewshot"
    )
    question_card("Success signals",
        how_to=["Agreement with coder, stability under paraphrase, consistent rationales."],
        note_key="s4_success", reviewed_key="s4_success"
    )
    st.info("**Carry forward:** task, inputs, output schema, guardrails, few-shots, success criteria.")

def render_step5():
    st.subheader("Step 5 — Annotation with Script")
    st.caption("Goal: reliable, transparent labels.")
    question_card("Annotation schema",
        how_to=["List labels/dimensions; define each + inclusion/exclusion notes."],
        examples=["`responsibility`: assigns causal/solution agency to gov/industry/citizens"],
        note_key="s5_schema", reviewed_key="s5_schema"
    )
    question_card("Storage format",
        how_to=["Choose JSONL/CSV/DB; include doc_id, span_id, label, rationale, annotator, timestamp."],
        note_key="s5_storage", reviewed_key="s5_storage"
    )
    question_card("Agreement & QC",
        how_to=["Pilot (e.g., 50 docs), double-code 10–20%, compute agreement or LLM self-consistency."],
        examples=["Duplicate items with paraphrase; gold examples."],
        note_key="s5_qc", reviewed_key="s5_qc"
    )
    question_card("Automation hooks",
        how_to=["Pre-processing (dedupe, language filter), batch prompting, retry-on-uncertain, dashboards."],
        note_key="s5_automation", reviewed_key="s5_automation"
    )
    question_card("Feedback loop",
        how_to=["Use annotated data to refine prompts/definitions; keep a changelog."],
        note_key="s5_feedback", reviewed_key="s5_feedback"
    )
    st.info("**Carry forward:** schema, storage plan, QC plan, automation hooks, feedback loop.")

def render_guide_tab():
    _init_guide_state()
    st.markdown("### Guided Walkthrough (read-and-go)")
    # Free navigation across 5 steps; no gating.
    st.segmented_control(
        "Steps",
        options=[1,2,3,4,5],
        format_func=lambda i: f"{i}",
        key="guide_step_selector"
    )
    st.session_state["guide_state"]["step"] = st.session_state["guide_step_selector"]
    step = st.session_state["guide_state"]["step"]

    if step == 1: render_step1()
    elif step == 2: render_step2()
    elif step == 3: render_step3()
    elif step == 4: render_step4()
    else: render_step5()

    # Sidebar mini progress (informational)
    with st.sidebar:
        st.divider()
        st.markdown("## Guide progress")
        reviewed = st.session_state["guide_state"]["reviewed"]
        for i, label in enumerate(["Topic", "Question", "Characters", "Prompt", "Annotation"], start=1):
            tick = "✅" if any(k.startswith(f"s{i}_") for k in reviewed) else "⬜️"
            st.write(f"{tick} Step {i} — {label}")
        st.caption("Notes & reviewed flags are optional.")

# ======================= TABS =======================
# NOTE: tabs are now 1) Guide, 2) Ask, 3) Playground
tab_guide, tab_ask, tab_play = st.tabs(["Guide", "Ask about the paper/repo (Local or OpenAI)", "Prompt playground"])

# ---------------- TAB 1: GUIDE (NEW) ----------------
with tab_guide:
    render_guide_tab()

# ---------------- TAB 2: Always-helpful Q&A with routing + memory ----------------
with tab_ask:
    if not stores:
        st.info("No documents indexed. Add files under data/sections, data/faq, data/guides, then Rerun.")
    else:
        if "chat" not in st.session_state:
            st.session_state.chat = []  # [{"role":"user"/"assistant","content": "..."}]

        llm = get_llm(provider, user_key, local_model)
        st.caption("Ask anything. The assistant uses sections/FAQ/guide when helpful; otherwise answers from general knowledge.")
        q = st.text_input("Your question", key="qa_question_input")

        clear = st.button("Clear chat")
        if clear:
            st.session_state.chat = []

        if q:
            # ROUTE to best corpus
            corpus, results = best_corpus_for_question(stores, q)
            ctx = "\n\n".join(d.page_content for d, _ in results) if results else ""

            # MEMORY: include last few turns for context
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

# ---------------- TAB 3: Prompt playground ----------------
with tab_play:
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

