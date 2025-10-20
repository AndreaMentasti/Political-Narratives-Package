import streamlit as st

# ───────────────────────── Page header ─────────────────────────
st.set_page_config(page_title="Political Narratives Guide", layout="wide")
st.title("Political Narratives Guide")

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
        """
    )

def render_step(step: int):
    """
    Content for each of the 5 steps:
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
            key_prefix="s2_example"
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

        question_card(
            "Guide: character selection ✅",
            how_to=[
                "Identify relevant characters for the topic (human and instrument) — this is the core of Step 3.",
                "Anchor selection in your research question and analytical focus; choose characters that speak to your hypotheses.",
                "Balance scope with feasibility: too many characters can reduce precision and raise compute costs; a focused set improves reliability and interpretability.",
                "Build the character list via literature review, exploratory tools (topic modeling, entity recognition/RELATIO), and domain reading; document your choices."
            ],
            ask_yourself=[
                "Which entities matter most for the topic at hand?",
                "Are these best understood as human actors (individuals, groups, organizations, states) or instrumental actors (policies, tools, institutions)?",
                "What is the scope of my analysis (national, regional, global)? Which entities fall outside my scope?",
                "Which characters recur most often in prior literature or theory on this topic?",
                "Do exploratory tools (topic models, word clouds, entity recognition, RELATIO outputs) highlight additional frequently mentioned entities?",
                "How many characters can I realistically track while keeping the coding interpretable and statistically useful?",
                "Is the chosen character list feasible for LLM coding (not too long, not too ambiguous)?",
                "Can each character plausibly appear in different roles (hero, villain, victim), or are some inherently neutral/instrumental?",
            ],
            key_prefix="s3_chars"
        )

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

        question_card(
            "Guide: prepare the prompt(s) ✅",
            how_to=[
                "Choose ONE main task (classify / extract / summarize / compare / generate) and one input unit (headline / paragraph / article / tweet).",
                "Define a **JSON schema** (keys, allowed labels, brief rationale) and **guardrails** (e.g., cite spans, no external knowledge, be concise).",
                "Add **2–4 worked examples** (cover easy + tricky edge cases).",
                "Co-design prompts with the same model you will use for annotation to align capabilities and outputs.",
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

        output_card(
            "What you should have before Step 5 ⚠️",
            bullets=[
                "A finalized **prompt** for both tasks (relevance + character/role), including schema and guardrails.",
                "A set of **few-shot examples** (both positive and near-miss).",
                "Consistent key names to enable parsing into (M, R) downstream."
            ],
            key_prefix="s4_output"
        )

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

        question_card(
            "Guide - GPT Annotation with provided code ✅",
            how_to=[
                "After the design of the prompts, it's time to run your annotation code against the chosen model.",
                "Inputs required:",
                "1) A dataset with observation id and text snippets.",
                "2) The two prompts, one for each task for the model.",
                "Before running, check your local folder structure and configuration."
            ],
            ask_yourself=[
                "Is my dataset ready for the annotation (id, text, missing values, etc)?",
                "Are my prompts correctly specified and specific enough?",
                "Is my folder structure organised as expected?"
            ],
            key_prefix="s5_outputs"
        )

        example_card(
            "Annotation output and relevant tweets 💡",
            (
                "A typical tidy output includes:\n"
                "- Stage-1 flags (relevance of the text)\n"
                "- Character presence (which characters appear)\n"
                "- Role indicators (whether a character is assigned a role)\n\n"
                "A text is a **political narrative** if at least one role assignment is present; otherwise it may still be relevant but neutral."
            ),
            key_prefix="s5_annotation_example"
        )

        output_card(
            "What you should have at the end 🎆",
            bullets=[
                "A **clean annotations file** in .csv (or JSONL) matching your schema. 🎆",
            ],
            key_prefix="s5_output"
        )

        st.text_area("Annotations for Step 5 (optional)", key="notes_s5",
                     value=st.session_state["guide"]["notes"][5], height=120)
        st.session_state["guide"]["notes"][5] = st.session_state["notes_s5"]

def render_guide_page():
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

# ───────────────────────── Single-page App ─────────────────────────
render_guide_page()



