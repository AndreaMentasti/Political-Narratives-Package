import streamlit as st

# ───────────────────────── Page header ─────────────────────────
st.set_page_config(page_title="Political Narratives Guide", layout="wide")
st.title("Political Narratives Guide")

# ───────────────────────── Helpers (Guide page) ─────────────────────────
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

def question_card(
    title: str,
    how_to: list[str],
    ask_yourself: list[str],
    key_prefix: str,
    blurb_md: str | None = None,
    figure: dict | None = None
):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        # NEW: short description just under the title
        if blurb_md:
            st.markdown(blurb_md)

        # NEW: optional figure (local path or URL)
        if figure and figure.get("src"):
            st.image(
                figure["src"],
                caption=figure.get("caption", ""),
                use_container_width=True
            )

        if how_to:
            st.markdown("**How to approach**")
            st.markdown("\n".join([f"- {p}" for p in how_to]))
        if ask_yourself:
            st.markdown("**Ask yourself**")
            for i, q in enumerate(ask_yourself, start=1):
                cb_key = f"{key_prefix}_q{i}"
                st.session_state["guide"]["registry"][cb_key] = True
                checked = st.checkbox(
                    q,
                    key=cb_key,
                    value=st.session_state["guide"]["done"].get(cb_key, False)
                )
                st.session_state["guide"]["done"][cb_key] = checked

def example_card(title: str, example_md: str, key_prefix: str, figure: dict | None = None):
    """A simple markdown example block (no checkboxes)."""
    with st.container(border=True):
        st.markdown(f"**Example — {title}**")
        # NEW: optional figure at the top of the example card
        if figure and figure.get("src"):
            st.image(
                figure["src"],
                caption=figure.get("caption", ""),
                use_container_width=True
            )
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

*A political narrative is identified by (i) its topic, (ii) its characters, and (iii) by having at
least one character cast in a drama triangle role: hero, villain, or victim.*

The definition and measurement of political narratives,  therefore, reduce to specifying the topic and characters, and coding for each character whether
it appears as neutral or cast as hero, villain, or victim.
Its purpose is influencing perceptions, beliefs, and preferences about characters contained in the narrative.  Political narratives exert their influence by depicting characters in one of the three archetypal roles—**hero**, **villain**, or **victim**.  They are communicative devices that focus attention, encode roles and identities, and shape norms and behavior.

Formally, fix a topic *T* and a universe of characters *K = H ∪ I*. For any text unit (tweet, paragraph, article), let *K′ ⊆ K* be the set of characters that appear.  
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
                "Is it likely there are enough identifiable characters within those narratives?",
                "Which data sources are most informative for this topic, and do I have reliable access to them?",
                "If those sources are available, can I obtain the essential metadata (dates, outlets, geography, language) needed for analysis?",
                "Is the research question compelling and relevant to the scientific community (and/or practitioners)?",
                "Could any actors or communities be harmed by this analysis, and how will I mitigate that risk?",
                "Is the topic sufficiently specific to be analyzable, without being so narrow that it lacks variation?"
            ],
            key_prefix="s1_scope"
        )

        example_card(
            "Focusing on policy narratives within climate change 💡",
            (
                "In *Gehring & Grigoletto (2025)* we analyze the **political economy of climate change**. "
                "From the literature we identify two dominant discussions—**scientific evidence** and **policy responses**—and, "
                "given our focus on political economy, we restrict attention to the topic of **climate change policies**. We explicitly exclude "
                "debates on the scientific reality and predictability of climate change."
            ),
            key_prefix="s1_example"
        )

        output_card(
            "What you should have before Step 2 ⚠️",
            bullets=[
                "A precise topic for the analysis of Political Narratives"
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

        question_card(
            "Guide: source selection & data extraction ✅",
            how_to=[
                "After selecting the topic, the next step is gathering data. Common sources include digitized newspapers, social media, transcribed TV/radio/YouTube content, and open-ended survey responses.",
                "When selecting the data source, prioritize the media channels where narratives about your chosen topic are most prominent.",
                "Evaluate trade-offs between coverage, accessibility, and quality (e.g., digitization errors, platform bias, sampling limits).",
                "For data extraction, the chosen source will determine which methodologies can be applied—such as keyword-based queries, scraping, API pulls, or manual collection.",
                "Consider the level of metadata you can preserve (dates, outlets, authors, geography, language) since these details will later be the basis of your analysis.",
                "What is the unit of analysis? Should I split the texts into smaller snippets, or can I work with the extracted texts as they are?"
            ],
            ask_yourself=[
                "Do the chosen sources capture the main media where the political debate unfolds?",
                "Are they sufficiently diverse to avoid bias toward one outlet, ideology, or demographic?",
                "Do I have legal and technical access to these data (e.g., archives, APIs, scraping permission)?",
                "What extraction method is most reliable for my source—keyword queries, metadata filters, or transcript parsing?",
                "Is the extraction method able to produce snippets that are neither too short to lose context nor too long to become too complicated for the analysis? If needed, split the texts into smaller snippets.",
                "Is the time window covered by the source appropriate for the research question?",
                "Can I obtain essential metadata (dates, outlets, geography, language) for contextual analysis?"
            ],
            key_prefix="s2_sources"
        )

        example_card(
            "Extracting data using Twitter Historical APIv2 💡",
            (
                "In *Gehring & Grigoletto (2025)* our focus is on narratives about climate change policies in the United States, collected from the social media platform Twitter. "
                "We specifically choose the U.S. due to the significant role Twitter plays in shaping and disseminating political narratives there. "
                "The data collection process involves querying the **Twitter historical APIv2** with a set of **keywords** adapted from *Oehl, Schaffer, and Bernauer (2017)*. "
                "In our main analysis, we define the tweet as our unit of observation, since its concise length aligns well with the requirements of the subsequent stages of the framework, including the GPT annotation process. "
                "We also extract newspaper articles, which we decided to split into smaller snippets to make them compatible with the framework’s next steps."
            ),
            key_prefix="s2_example"
        )

        output_card(
            "What you should have before Step 3 ⚠️",
            bullets=[
                "A dataset with the extracted text snippets and other metadata, if you need them for your analysis."
            ],
            body_md="""
        | id   | text               | ...
        |------|--------------------|----
        | 1_1  | If you listen…     |      
        | 1_2  | But is it po…      |      
        | 1_3  | In fact, on…       |      
        """,
            key_prefix="s2_output"
        )

    # --- STEP 3 ---
    if step == 3:
            st.subheader("Step 3 — Identify relevant characters")
            st.caption("Map the topic into human and instrument actors with agency and claims.")

            question_card(
                "Guide: character selection ✅",
                how_to=[
                    "Identify relevant characters for the topic — this is the core of Step 3.",
                    "Anchor selection in your research question and analytical focus; choose characters that speak to your hypotheses.",
                    "Balance scope with feasibility: too many characters can reduce precision and raise compute costs; a focused set improves reliability and interpretability. The list of characters doesn't need to include the whole universe of character relevant for the topic, it just needs to be consistent with the scope of your research.",
                    "Build the character list via literature review, exploratory tools (topic modeling, entity recognition/RELATIO), and domain reading; document your choices."
                ],
                ask_yourself=[
                    "Which characters matter most for the topic at hand?",
                    "What is the scope of my analysis (national, regional, global)? Which characters fall outside my scope?",
                    "Which characters recur most often in prior literature or theory on this topic?",
                    "Do exploratory tools (topic models, word clouds, entity recognition, RELATIO outputs) highlight additional entities? If so, how to aggregate these entities into some broader character definitions?",
                    "Do the selected characters exhibit distinctions that are sufficiently clear for an LLM to recognize and differentiate them?",
                    "Is the chosen character list feasible for LLM coding (not too long, not too ambiguous)?",
                    "If interested in human vs instrument classification, are these best understood as human actors (individuals, groups, organizations, states) or instrumental actors (policies, tools, institutions)?",
                    "How many characters can I realistically track while keeping the coding interpretable and statistically useful?",
                    "Can each character plausibly appear in different roles (hero, villain, victim), or are some inherently neutral?",
                ],
                key_prefix="s3_chars",
                # ↓ NEW: short description appears above “How to approach”
                blurb_md=(
                    "Define a **small, distinctive set of characters** that directly reflect your research question. "
                    "Prefer **clear, non-overlapping definitions** that an LLM can reliably identify across texts. "
                    "Record a **brief description for each character now**—you will reuse it in Step 4 prompts."
                ),
            )

            example_card(
                "Relevant characters for climate change political discourse 💡",
                (
                    "**Text Example**\n\n"
                    "“Global greenhouse emissions are still on the rise, oil production is soaring and **energy companies** "
                    "are making sky-high profits while **countless people** struggle to pay their bills. [...] A critical mass of "
                    "people – especially **younger people** – are demanding change and will no longer tolerate the "
                    "procrastination, denial and complacency that created this state of emergency.”\n\n"
                    "In the text and figure above it is clear how the characters can be seen as nodes of DAGs. However, the diagram shows that assigning causal arrows between characters may often be ambiguous in real texts. "
                    "By contrast, assigning roles is typically clearer and can be coded directly:  in this example, corporations are cast as villain, the poor as victim, and civil society as hero.\n\n"
                    "Guided by the relevant literature, exploratory tools, and intensive domain reading, "
                    "we pre-specify ten characters: five human characters (made of institutions and groups of individuals) "
                    "and five instrument characters (policy tools and instruments).\n\n"
                    "**Human Characters:** DEVELOPING ECONOMIES | US DEMOCRATS | US REPUBLICANS | CORPORATIONS | US PEOPLE\n\n"
                    "**Instrument Characters:** EMISSION PRICING | REGULATIONS | FOSSIL INDUSTRY | GREEN TECH | NUCLEAR TECH\n\n"
                    "We carefully decided to have ten character to balance the complexity of the analysis with "
                    "good practices to avoid overloading the LLM in the annotation process. These characters are easily recognizable "
                    "by the LLM thanks to precise descriptions in the prompt design step (Step 4)."
                ),
                key_prefix="s3_example",
                figure={
                    "src": "assets/dag_git.png",  # path or URL to your image
                    "caption": "Characters as Nodes."
                }
            )

            output_card(
                "Output - What you should have before Step 4 ⚠️",
                bullets=[
                    "A contained list of relevant characters, both in human and instrument form if needed",
                    "A brief description of each character should be included in the prompts at Stage 4. It is good practice to annotate these descriptions while selecting the characters."
                ],
                key_prefix="s3_output"
            )

            st.text_area("Annotations for Step 3 (optional)", key="notes_s3",
                         value=st.session_state["guide"]["notes"][3], height=120)
            st.session_state["guide"]["notes"][3] = st.session_state["notes_s3"]

    # --- STEP 4 (restored) ---
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
                "Output JSON contains keys a–j (one per character)."
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
            "Annotation output and relevant texts 💡",
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




