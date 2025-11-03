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
The purpose of a political narrative is influencing perceptions, beliefs, and preferences. Political narratives exert their influence by depicting characters - human or instrument - in one of the three archetypal roles — **hero**, **villain**, or **victim**. They are communicative devices that focus attention, encode roles and identities, and shape norms and behavior.

We can also depict this formally. To define and measure narratives, you need to choose a topic T and a universe of characters K = H ∪ I, where H and I represent Human and Instrument characters. For any text unit (tweet, paragraph, article), let K′ ⊆ K be the set of characters that appear.  
A role-assignment function *r : K′ → {hero, villain, victim, neutral}* maps each appearing character to either a drama-triangle role or neutrality. We call *(T, K′, r)* a **political narrative** if and only if at least one character is cast as hero, villain, or victim. If all characters are neutral, the text is about the topic but does not constitute a political narrative in this sense.  

**How to use this guide**
- Use the step selector above to move from **1 → 5**.
- Each step includes three cards:
  - **Guide ✅** — brief “How to” plus reflective **Ask yourself** items.
  - **Example 💡** — a concrete mini-case clarifying the step.
  - **Output ⚠️** — what you should have before moving on.
- Jot ideas in the **Notes** box at the end of each step and keep notes on your progress.
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
                "restrict analysis to a manageable set. Clearly document what aspects of the topic are included and excluded from the analysis."
            ],
            ask_yourself=[
                "Is the topic sufficiently specific to be analyzable, without being so narrow that it lacks variation?",
                "Does this topic surface enough distinct political narratives and public debate to analyze?",
                "Is it likely there are enough identifiable characters within those narratives? Characters can be individuals, parties, institutions, or groups appearing in the discourse.",
                "Is the research question compelling and relevant to the scientific community (and/or practitioners)?",
                "Which data sources are most informative for this topic, and do you have reliable access to them?",
                "If those sources are available, can you obtain the essential metadata (dates, outlets, geography, language) needed for analysis?",
                "Could any actors or communities be harmed by this analysis, and how would you mitigate that risk?"
            ],
            key_prefix="s1_scope"
        )

        example_card(
            "Focusing on policy narratives within climate change 💡",
            (
                "In *Gehring & Grigoletto (2025)* we analyze the **political economy of climate change**. From the literature and reading our potential sources, we realized there are two rather distinct political discussion – one about the scientific evidence behind the causes and consequences of climate change, one about policies to address climate change.  "
                "Given our research interest is on the political reactions and discourse about political solutions, we define our topic more narrowly as **climate change policies**. We explicitly exclude debates on the scientific reality and predictability of climate change."
            ),
            key_prefix="s1_example"
        )

        output_card(
            "What you should have before Step 2 ⚠️",
            bullets=[
                "A precise topic for the analysis of political narratives. We suggest writing it up in full sentences, but also trying to define it via bullet points. Both a positive definition is useful (what should be included), as well as a negative distinction (e.g. it should NOT include certain related or overlapping topics). This topic defines where to look and what kind of material to collect, hence is the basis for Step 2 Sources and Extraction"
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
                "What is the unit of analysis (for instance, a tweet, paragraph, or newspaper snippet)? Should I split the texts into smaller snippets, or can I work with the extracted texts as they are? For example, long texts (e.g., articles) can be segmented into paragraphs or 3-sentence chunks to fit LLM input limits and maintain focus.",
                "If text snippets are too short, you will not capture more complex narratives with multiple character-roles, but mostly narrative fragments. If it is too long, the LLM will struggle to assign a role to a character with our current prompts, for instance if the character is portrayed in different ways in a long article. For example, long texts (e.g., long newspaper articles or TV segments) should be segmented into smaller text snippets. From experience, something along the length of a single tweet, a paragraph or 3-sentence chunks are a useful compromises. If the text is too short, you will not capture more complex narratives with multiple character-roles, but mostly narrative fragments. If it is too long, the LLM will struggle to assign a role to a character with our current prompts. Generally we advise users to pick units that have a natural meaning in an article, .e.g. a sentence or a paragraph usually follow a structure, whereas taking a certain number of words before and after a keyword breaks sentence and characters that might have had a meaning.",
                "If very long texts cannot be avoided, one should consider specifying in the prompts that in Step 4 how to cope with contradictory or changing roles over the course of the text"
            ],
            ask_yourself=[
                "Do the chosen sources capture the main media where the political debate unfolds?",
                "Is the time window covered by the source appropriate for the research question?",
                "Are they sufficiently diverse to avoid bias toward one outlet, ideology, or demographic?",
                "Do you have legal and technical access to these data (e.g., archives, APIs, scraping permission)? Ensure compliance with copyright and platform terms when extracting data.",
                "What extraction method is most reliable for your source—keyword queries, metadata filters, or transcript parsing?",
                "Is the extraction method able to produce snippets that are neither too short to lose context nor too long to become too complicated for the analysis? If needed, split the texts into smaller snippets.",
                "Basic preprocessing (removing duplicates, non-textual artifacts) ensures better results in later steps. If texts are multilingual, is good to consider filtering or adding a ‘language’ column for clarity."
            ],
            key_prefix="s2_sources"
        )

        example_card(
            "Extracting data using Twitter Historical APIv2 💡",
            (
                "In *Gehring & Grigoletto (2025)* our focus is on narratives about climate change policies in the United States, collected from the social media platform Twitter. "
                "We specifically choose the U.S. due to the significant role Twitter plays in shaping and disseminating political narratives there. "
                "The data collection process involves querying the **Twitter historical APIv2** with a set of **keywords** adapted from *Oehl, Schaffer, and Bernauer (2017)*. "
                "In our main analysis, we define the tweet as our unit of observation, since its concise length aligns well with the requirements of the subsequent stages of the framework, including the GPT prediction process. "
                "We also extract newspaper articles and TV transcripts, which we decided to split into smaller snippets to make them compatible with the framework’s next steps.\n\n"
                "- **Tweets:** Tweets have been extracted through a keyword based search using the historical API of Twitter\n\n"
                "- **Newspaper Articles:** Newspaper articles are downloaded from Factiva. We use the three most widely circulated newspapers in the US; The New York Times, The Wall Street Journal, and USA Today. Each article has been split into more manageable text snippets (three sentences paragraphs) for OpenAI classification.\n\n"
                "- **TV Transcripts:** TV transcripts have been downloaded from the GDELT database. These TV transcripts from MSNBC and Fox News have been also split into smaller text snippets (three sentences paragraphs)."
            ),
            key_prefix="s2_example"
        )

        output_card(
            "What you should have before Step 3 ⚠️",
            bullets=[
                "A dataset with the extracted text snippets and other metadata, if you need them for your analysis.",
                "Store your extracted data in a UTF-8 CSV with at least two columns: id and text, plus optional metadata (date, outlet, etc.)."
            ],
            body_md="""
        | id   | text               |...
        |------|--------------------|---
        | 1_1  | If you listen…     |...   
        | 1_2  | But is it po…      |...
        | 1_3  | In fact, on…       |...   
        """,
            key_prefix="s2_output"
        )

        st.text_area("Annotations for Step 2 (optional)", key="notes_s2",
                     value=st.session_state["guide"]["notes"][2], height=120)
        st.session_state["guide"]["notes"][2] = st.session_state["notes_s2"]

    # --- STEP 3 ---
    if step == 3:
            st.subheader("Step 3 — Identify relevant characters")
            st.caption("Map the topic into relevant human and instrument characters.")

            question_card(
                "Guide: character selection ✅",
                how_to=[
                    "Identify relevant characters for the topic. The basis for the selection can be the relevant literature, your own interests as a researcher, or be approached in a more data driven way. You can identify characters manually by reading and noting recurring entities. Automated tools can help, but they’re optional. ",
                    "As said above, each **character** should be seen as a category and not be mistaken with *actors*. The necessary first step is domain reading: you need to read the text snippets that you plan to encode. The tasks of the LLM later is to encode at scale, but any choices should also be plausible for a human reader. ",
                    "Traditional tools such as word clouds or topic models can help. You may also use entity recognition with the more advanced [RELATIO package](https://github.com/relatio-nlp/relatio). However, all these tools require a dimensions reduction at some point. From the full set of possible characters, you need to decide how to aggregate to a feasible and usable number of chosen characters. In contrast to topic models or RELATIO, this choice comes before the encoding at scale. ",
                    "Entity recognition is very useful for thinking about the necessary aggregation, as it can tell you which entities appear how often. This provides a good basis for thinking which entities should be aggregated into one character. For example, in real text the character **US president Trump** might appear as *Trump*, *Donald J. Trump*, *POTUS*, *US president*, etc. , but conceptually clearly represents just one character.",
                    "A different aggregation task is to cluster several narrower characters into a broader one. For instance, you could have three characters *solar technology*, *wind technology*, and *geothermic technology* if you are interested in understanding specific narratives about the specific technologies. Or you can decide to aggregate all up into one character **green technology**, if your goal is to capture the narrative roles of green technology as a combined category. Similar examples would be looking at *primary education*, *secondary education*, and *tertiary education* separately, or having one character **education**. Hence one can think of a character as an aggregation of different words for the same clearly defined character, or of a larger character as a cluster aggregating several more specific characters.",
                    "For a precise prediction, characters should be internally as homogenous as possible, and as heterogenous and clearly differentiated compared to other characters as possible",
                    "Balance scope with feasibility: too many characters can raise computational costs and prediction complexity; a focused set improves reliability and interpretability. However, aggregating too many entities into one character will make a prediction more noisy, unless the mapping into the character is always obvious and clear. It can make sense to start with a larger set of characters which could be aggregated later for regression analysis, but be aware it comes at these monetary costs and with a potential loss in precision.",
                    "Document your choices and motivations for later. Also for each character, it helps to have both a short positive list of the key entities or more specific characters that it includes, as well as potentially a negative distinction of things that it does not comprise.",
                    "Go back and read a sufficient number of text snippets manually to validate your choices.",
                    "Key is to anchor the selection to your research question and analytical focus; choose characters that speak to your hypotheses and that you want to use in your descriptives or regressions later.",
                    "The list of characters doesn't need to include the whole universe of characters relevant for the topic, it just needs to be consistent with the scope of your research."
                ],
                ask_yourself=[
                    "Which characters recur most often in prior literature or theory on this topic?",
                    "Which characters matter most for the topic at hand?",
                    "Do your characters appear in at least one drama triangle role (hero, villain or victim)? Or even in several? It can be interesting to observe if they only appear as neutral, but often more interesting if they can take on more roles",
                    "Do exploratory tools (topic models, word clouds, entity recognition, RELATIO outputs) highlight additional entities? If so, how to aggregate these entities into some broader character definitions?",
                    "What is the scope of your analysis (national, regional, global)? Which characters fall outside your scope?",
                    "Do the selected characters exhibit distinctions that are sufficiently clear for an LLM to recognize and differentiate them?"
                ],
                key_prefix="s3_chars",
                # ↓ NEW: short description appears above “How to approach”
                blurb_md=(
                    "With your data collected, you can now continue to define the characters within the scope of your topic. "
                    "Characters can be human (individuals or collective actors such as corporations, parties, states, movements) but also instruments/instrumental (policies, laws, technologies). "
                    "This means that the term **character** should be seen as a category and not be mistaken with *actors*; it does not need to be human or an individual, but can also be groups of people, policy areas or individual policies, or more abstract categories like science, a technology or technology class.\n\n "
                    "In this step, define a **small, distinctive set of characters** that directly reflect your research question. "
                    "Prefer **clear, non-overlapping definitions** that an LLM can reliably identify across texts. To manage this, ask whether an outsider could tell these characters apart from their descriptions alone. "
                    "Record a **brief description for each character now**—you will reuse it in Step 4 Prompts."
                ),
            )

            example_card(
                "Relevant characters for climate change political discourse 💡",
                (
                    "#### Political Narrative Characters for Climate Change Policy:\n\n"
                    "Guided by the relevant literature, exploratory tools, and intensive domain reading, "
                    "we pre-specify ten characters: five human characters (made of institutions and groups of individuals) "
                    "and five instrument characters (policy tools and instruments).\n\n"
                    "**Human Characters:** DEVELOPING ECONOMIES | US DEMOCRATS | US REPUBLICANS | CORPORATIONS | US PEOPLE\n\n"
                    "**Instrument Characters:** EMISSION PRICING | REGULATIONS | FOSSIL INDUSTRY | GREEN TECH | NUCLEAR TECH\n\n"
                    "We carefully decided to have ten character to balance the complexity of the analysis with "
                    "good practices to avoid overloading the LLM in the prediction process. These characters are easily recognizable "
                    "by the LLM thanks to precise descriptions that include positive examples and partly negative distinctions to related concepts. (Step 4).\n\n"
                    "##### Examples of Characters: \n\n"
                    "In our application, we were not interested in individual politicians, but in the distinction in democrats and republicans. The means one of our characters was **US Democrats**. This is then a cluster of the many different humans and the organization, e.g. members of parliament, executive members, governors, etc., but also the democratic party itself. "
                    "It also includes many different ways of referring to these more specific characters. We do not need to specify all of these, a couple of examples are sufficient for modern LLMs. "
                    "Another example is the character **Green Technologies**. We were not interested in specific technologies, so this contains *clean tech sector*, *bioenergy*, *solar energy*, and similar entities. We noted some examples of technologies to provide them to the LLM in the next stages.\n\n" 
                    "##### Characters as Nodes in DAG: \n\n"
                    "Global greenhouse emissions are still on the rise, oil production is soaring and **energy companies** "
                    "are making sky-high profits while **countless people** struggle to pay their bills. [...] A critical mass of "
                    "people – especially **younger people** – are demanding change and will no longer tolerate the "
                    "procrastination, denial and complacency that created this state of emergency.\n\n"
                    "In the text and in the Figure *Characters as Nodes* it is clear how the characters can be seen as nodes of DAGs. The different entities in the text are identified within broader characters: corporations (*energy companies*), the poor (*countless people*), and civil society (*younger people*).  However, the diagram shows that assigning causal arrows between characters may often be ambiguous in real texts. "
                    "By contrast, assigning roles is typically clearer and can be coded directly:  in this example, corporations are cast as villain, the poor as victim, and civil society as hero."
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
                    "A concise table listing each character and its short description, ready to be inserted into Step 4 prompts.",
                    "A brief description of each character should be included in the prompts at Stage 4. It is good practice to annotate these descriptions while selecting the characters. You can revisit and refine these descriptions once you see the prompt in Step 4."
                ],
                
                key_prefix="s3_output"
            )

            st.text_area("Annotations for Step 3 (optional)", key="notes_s3",
                         value=st.session_state["guide"]["notes"][3], height=120)
            st.session_state["guide"]["notes"][3] = st.session_state["notes_s3"]

    # --- STEP 4 (restored) ---
    if step == 4:
        st.subheader("Step 4 — Prepare the prompt(s)")
        st.caption("After defining your characters in Step 3, you now need to instruct the model on how to detect them. Step 4 translates your conceptual framework into an operational prompt that the LLM can understand.")

        # 1) GUIDE
        question_card(
            "Guide: prepare the prompt(s) ✅",
            how_to=[
                "Define a **JSON schema** (keys, allowed labels, brief explanation of the task).",
                "Co-design prompts with the same model you will use for prediction (e.g., GPT-4o-mini via OpenAI) to align capabilities and outputs.",
                "This package supports two tasks: (a) topic relevance classification and (b) character detection + role assignment. Choose which task perform, and consider applying prompt 1 to increase the quality of your data."
            ],
            ask_yourself=[
                "Is the task singular and clear (avoid mixing multiple tasks in one prompt)? Use the correct prompt for the task that you need to perform.",
                "Is the chosen input unit appropriate for context vs. speed constraints? Remember, shorter text snippets are handled better by the OpenAI API.",
                "Is the schema unambiguous, machine-readable, and easy to parse? Are the keys correctly specified and the roles defined properly?",
                "Do I provide a description of the selected characters?",
                "If you copy-paste one test snippet into the model with this prompt (Chat gpt), do I get valid JSON with all keys? (If not, fix BEFORE Step 5.).",
            ],
            key_prefix="s4_prompt",
            
            blurb_md=(
                "This step is crucial for achieving **accurate classification**. You must provide "
                "**precise instructions** to the LLM to ensure that characters are correctly annotated with their respective roles. "
                "To do so, you need to create a **SYSTEM MESSAGE** in a JSON file that clearly specifies three aspects: first, the task that the LLM has to perform on each text snippet; "
                "second, the set of characters to be identified, together with a brief description of them, and with clarifying examples; lastly, the set of possible roles they can assume, which in this case are set to Hero, Villain, and Victim.\n\n "
                "Below, we provide key guidance for constructing the SYSTEM MESSAGE. A ready-to-use example prompt is available "
                "in the [GitHub repository](https://github.com/AndreaMentasti/Political-Narratives-Package/tree/main) and can be easily adapted by following these instructions.\n\n "
                "To further improve data quality, we also include an optional **relevance classification** prompt. This further helps assessing how well the selected text snippets relate to the topic. Think of it this way : in Step 2, one usually searches Boolean logic with certain keywords and conditions to prefilter the text. However, this is a noisy process and might still leave many text snippets that are de facto not related to your topic of interest. "
                "This will introduce noise in your data, making both the prediction and later analysis more difficult. Depending on the extend of the issue, the relevance classificaiton can help you reduce that noise (and it saves cost and time in the classification). If being applied, a user needs to first run the relevance classification first (Stage 1), and should then apply the character-role classification (Stage 2) on those texts labeled as relevant for your topic by step 2 and the relevance classification.\n\n"
                "##### Step-by-Step instructions:\n"
                "1) Access the [GitHub repository](https://github.com/AndreaMentasti/Political-Narratives-Package/tree/main) and download the `system_message_stage1` for the relevance classification and the `system_message_stage2` for the character-role classification from the `code and prompts/` folder.\n\n "
                "2) For the relevance classification with `system_message_stage1`, you just need to adjust the instructions in the SYSTEM MESSAGE to ensure they reflect the particular topic under analysis. Nothing else is needed in this prompt.\n\n"
                "3) For the character-role classification with `system_message_stage2`, you need to modify the instructions, the list of characters, and their descriptions.\n\n"
                "**How to modify the two system messages?** In the prompts you will find the following text: *You are an average US citizen. The user will provide the content of a tweet posted from the US between 2010 and 2021. \nYour task is to analyze it within the context of US political discourse, particularly in relation to climate change and related policies.*\n\n "
                "This is an example used in our paper as we were interested in assessments by US citizens. Narratives are always context-dependent, and LLMs can at least to some extent capture this. We strongly suggest adapting this part to your target population. If it is about India, insert *You are an average Indian citizen*. Similar for other context. The more refined the better, as long as it is clearly defined. E.g. if you have five European countries, we suggest either running five separate prompts - if you feel the narratives and their assessment is very dependent on culture - or a single prompt if you specifically want an *average European perspective*. In the end, the researchers should elaborate and decide, but be aware that context matters. To adapt the prompts, you have to modify these few instructions with their specific task. Then, if you are working on the character-role classification you can continue following these instructions. For the relevance classification, you don't need other changes to the prompt and you are ready to proceed to Step 5.\n\n "
                "By following the next steps you will be able to modify the character keys by adding the topic-specific characters.\n\n"
                "4) Next, for `system_message_stage2` you can modify the names of the characters and their descriptions. In the example that we provide, we specified 10 character that are stored in the keys from a to j. Moreover, we included also descriptions and examples of each character. Paste your characters and descriptions in place of the examples in the prompt.\n\n"
                "5) After each description, you must specify in which key to store the result. The alphabetical keys are important to enter in non-capitalized form, always starting from a, then b, then c, running until X, which depends on the number of characters selected: with six characters, X=f, and the columns run from a-f. For example, if there are eight characters, the prompt will define keys from a to h. As a result, the classification output for the first character will be stored in column a (key a), for the second character in column b (key b), and so on. "
                "Keys simply indicate how the LLM stores the results according to the prompt. You do not need any knowledge of JSON syntax to modify these few letters in the prompt.\n\n"

            ),
        )

        # 2) EXAMPLE
        example_card(
            "Prompt design for the political narratives of climate change 💡",
            (
                "###### (a) Relevance classifier (Stage 1)\n"
                "Decide if a three-sentence tweet/newspaper/other text source excerpt is relevant to **US climate change discourse**.\n"
                "- 0 = irrelevant (no meaningful climate discussion)\n"
                "- 1 = assert (asserts existence of climate change)\n"
                "- 2 = deny (denies or mocks climate change)\n"
                "- 3 = relevant (substantive discussion of climate change policy)\n\n"
                "We used this script to filter the downloaded tweets and to keep only those that are actually relevant for the climate change policy discourse. This step can be seen as complementary to the topic selection step, meaning that this increases data quality and relevance to the topic. This is useful to avoid useless classification of irrelevant tweets and hence save money and time.\n\n"
                "###### (b) Character detection & role assignment (Stage 2)\n"
                "Identify pre-specified characters and assign contextual roles:\n"
                "- Villain (1) | Hero (2) | Victim (3) | No role (4)\n\n"
                "Characters: Developing Economies, US Democrats, US Republicans, Corporations, US People, "
                "Emission Pricing Tools, Regulation Policies, Fossil Fuels, Green Technologies, Nuclear Energy.\n"
                "Output JSON contains keys a–j (one per character).\n\n"
                "In the prompt, for each character we give a brief description and some positive examples to look for in the text."
            ),
            key_prefix="s4_example"
        )

        # 3) OUTPUT
        output_card(
            "What you should have before Step 5 ⚠️",
            bullets=[
                "One or two prompts, adapted to your topic, your location/country/culture and your characters. The prompt `system_message_stage1` for the relevance classification contains the system message with the instructions and the categories in which a text can be classified (coded from 0 to 3). The `system_message_stage2` contains the system message with the instructions, the list of characters that you selected in Step 3, and the description of each character.",
                "You now have both prompts ready. In the next step, you’ll run the provided code to generate your annotated dataset."
            ],
            key_prefix="s4_output"
        )

        # ——— PROMPT PASTE AREAS (stores in session_state) ———
        st.markdown("##### Examples of JSON prompts")
        st.caption("Provide machine-readable JSON specs for Stage 1 (relevance classification) and Stage 2 (character/role assignment). You can find the original prompts for *Gehring and Grigoletto (2025)* in the GitHub repository.")

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
        # --- initialize once ---
        if "s4_relevance_json" not in st.session_state:
            st.session_state["s4_relevance_json"] = default_relevance_json

        if "s4_roles_json" not in st.session_state:
            st.session_state["s4_roles_json"] = default_roles_json

        st.text_area(
            "🧩 (a) Stage 1 — Relevance prompt JSON",
            key="s4_relevance_json",
            value=st.session_state["s4_relevance_json"],
            height=320
        )

        st.text_area(
            "🧩 (b) Stage 2 — Character & role prompt JSON",
            key="s4_roles_json",
            value=st.session_state["s4_roles_json"],
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
        st.caption("Run prediction, parse JSON, and build tidy outputs (stage flags, presence, role dummies).")

        question_card(
            "Guide - GPT prediction with provided code ✅",
            how_to=[
                "After the design of the prompts, it's time to run the prediction code against the chosen model. The following two inputs are required:",
                "1) A dataset with observation id and text snippets.",
                "2) The two prompts, one for each task for the model.",
                "Before running, check the local folder structure and configuration.",
                "Choose which script to run based on the task that is performed (relevance or character-roles classification)."
            ],
            ask_yourself=[
                "Is your dataset ready for the prediction (id, text, missing values, etc)?",
                "Does the dataset contain a column called ''id'' and a column called ''text''?",
                "Are your prompts correctly specified and specific enough?",
                "Is your folder structure organised as expected? To check this point we suggest you to visit the github repository and the instructions listed there.",
                "Are you in the correct environment (Political_Narratives)?",
                "Is the OpenAI API key set in the environment?"
            ],
            key_prefix="s5_outputs",

            blurb_md=r"""
                This is the final step of the pipeline. You will now run the prediction code that applies your prompts to your dataset. 
                This page of the guide will help you adapting the scripts that we provide in the Github repository to your objective. 
                In practice, this step is where you will apply the model instructions finalized in Step 4 to your text data, producing the full narrative prediction.\n
                ##### Step-by-Step instructions:\n
                1) **(For beginner Python users)** The first step is to download Python at the [link](https://www.python.org/downloads/).
                2) **(For beginner Python users)** Then, we suggest you to download Anaconda and Anaconda Prompt at the [link](https://www.anaconda.com/docs/getting-started/anaconda/install).
                3) Download the files in *code and prompts* and the *environment.yml*. Then, build the following folder structure locally:
                ```bash
                <main>/
                ├─ code/
                │  ├─ annotation_openai_stage1.py
                │  ├─ annotation_openai_stage2.py
                │  └─ prompts/
                │     ├─ system_message_stage1.json        # <-- your input message
                │     └─ system_message_stage2.json        # <-- your input message
                ├─ data/
                │  └─ output/
                │     ├─ your_data_stage1.csv              # <-- your input CSV
                │     └─ your_data_stage2.csv              # <-- your input CSV
                ├─ output/
                │  └─ data/
                │     ├─ openai_output/
                │     └─ openai_final/
                └─ logs/    
                └─ environment.yml                         # <-- your environment requirement
                ```
                According to this structure, you need to fill the `code/prompts/` folder with the system message of the stage, and the `data/output/` folder with the `.csv` dataset. Put the `environment.yml` in the main project folder.  
                IMPORTANT: the input `.csv` must contain a column called `id` and a column called `text`.
                
                4) Make sure your finalized prompt files (`system_message_stage1.json` and `system_message_stage2.json`) are saved in the correct folder, as the scripts will load them automatically.
                4) Make minimal changes in the code to match your path, the characters defined in the prompt, and the dataset. If you want to perform relevance classification, you just need to change the path.\n\n
                All the steps are explained in detail [here](https://github.com/AndreaMentasti/Political-Narratives-Package/tree/main). Check the detailed GitHub instructions if you need guidance."""
                
        )

        example_card(
            "Prediction output and relevant texts 💡",
            (
                "In the paper, our output is a dataset where each column corresponds to a character (a–X) and contains numeric role codes: 1 = Villain, 2 = Hero, 3 = Victim, 4 = No role:\n"
                "- Character presence (which characters appear)\n"
                "- Role indicators (whether a character is assigned a role)\n\n"
                "A text is defined as a **political narrative** if at least one characters has an assignment of a drama triangle role; if all are neutral we do not categorize it as a narrative."
            ),
            key_prefix="s5_annotation_example"
        )

        output_card(
            "What you should have at the end 🎆",
            bullets=[
                "A dataset with your text snippets as the rows, with the columns indicating your characters, and the cell values indicating the roles. X depends on the number of characters selected: with six characters, X=f, and the columns run from a-f. At this point your dataset is fully annotated and ready for analysis or visualization: 🎆",
            ],
            body_md="""
                | id   | text               | a  | b  | c  | d  | e | f |...|X
                |------|--------------------|----|----|----|----|----|----|----|---
                | 1_1  | If you listen…     |1   | 4  |0   | 0  | 0  | 0  |    |  0
                | 1_2  | But is it po…      |0   | 4  |0   |2   | 2  | 0  |    |  0
                | 1_3  | In fact, on…       |3   | 1  |0   | 0  | 0  | 0  |    |  0 
                """,
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




