# Political Narratives Package

The **Political Narratives** Package allows users to adapt the Political Narratives framework presented in [Gehring & Grigoletto (2025)](https://www.econstor.eu/handle/10419/327674) to their own research. By following the steps described in this repository, you will be able to systematically identify political narratives in your own text data. 

### What is a Political Narrative? 
According to [Gehring & Grigoletto (2025)](https://www.econstor.eu/handle/10419/327674), a *political narrative* is defined by three elements: *(i) a topic, (ii) a set of characters, and (iii) at least one character cast in a drama triangle role: hero, villain, or victim*. 
These roles, rooted in classic storytelling traditions, provide a simple and powerful structure that has shaped narratives since the earliest written accounts. Political narratives act as interpretive tools: they simplify complexity, assign meaning, and guide perception. By casting characters into familiar archetypes, they help audiences make sense of political issues and events, shaping beliefs, attitudes, and preferences in the process.

More formally, let *T* be a topic and *K = H ∪ I* a universe of characters, where H and I represent Human and Instrument characters. For any text unit (tweet, paragraph, article), let *K′ ⊆ K* be the set of characters that appear.  
A role-assignment function *r : K′ → {hero, villain, victim, neutral}* maps each appearing character to either a drama-triangle role or neutrality. We call *(T, K′, r)* a **political narrative** if and only if at least one character is cast as hero, villain, or victim. If all characters are neutral, the text is about the topic but does not constitute a political narrative in this sense.

### What Can You Expect from This Repository?
This package is meant to support you from start to finish: guiding you from the very first step of selecting a topic, all the way to generating your final dataset. It provides everything you need: the code to query the OpenAI API, prompts you can use or adapt, and clear guidelines to help you apply the Political Narrative framework to your own data. To help you in this process, we created an [interactive online guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/), taking you through the five steps of our proposed pipeline.

### How Should You Proceed?
1. **Read the reference paper**, [Gehring & Grigoletto (2025)](https://www.econstor.eu/handle/10419/327674):
   Focus especially on Sections 1 and 2: the introduction, which contextualizes the Political Narrative framework, and the section that defines and illustrates political narratives with examples.

2. **Open the [interactive online guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/)**:
   Let us walk you through each step of the pipeline, from selecting a topic to generating your final dataset.

3. **Adapt the material to your own project**:
   Use the code, prompts, and instructions we provide to prepare your first dataset of political narratives.

⚠️ *Note*: The online guide may take a couple of minutes to load the first time you open it.  

⚠️ *Working offline?* You can download a full offline version of the guide here: **[LINK TO PDF]**

---

### Repository Structure
- ````codes and prompts\````: this is the core of the repository. Here you can download the Python scripts and the prompts needed to perform the Political Narrative analysis. Moreover, here you can find the `environment.yml` file to create the environment where to run the scripts.
- ````useful_resources\````: here is contained the paper by *Gehring & Grigoletto (2025)*. You can access it and get a deeper understanding on how to shape a research using the framework. Moreover, here you find an offline version of the instructions to adapt the scripts provided in ````codes and prompts\````. This guide is exactly Step 5 Coding in the online guide.

---

## Inputs - Output Map
Inputs:
- dataset with text snippets and observation id (in the code ```your_code_stage1.csv``` and ```your_code_stage2.csv```)
- prompt for stage 1 (in the code ```system_message_stage1.json```)
- prompt for stage 2 (in the code ```system_message_stage2.json```)
- API key as Environmental variable
  
Outputs:
- Once run the code, you will find the output dataset in the folder ````output/data/openai_final/```` in a folder called as the date and time when the code has been run. The dataset is saved as ````.csv```` named ````flattened_results_all.csv````.

- Dataset with characther-role flags (if you run the second script):
  
  | id   | text               | dev | dem | rep | corp | ppl | pric | ban | fos | green | nuc |
  |------|--------------------|-----|-----|-----|------|-----|------|-----|-----|-------|-----|
  | 1_1  | If you listen…     |  0  |  0  |  2  |  0   |  0  |  0   |  0  |  1  |   0   |  0  |
  | 1_2  | But is it po…      |  0  |  0  |  0  |  0   |  0  |  0   |  0  |  0  |   0   |  0  |
  | 1_3  | In fact, on…       |  0  |  1  |  0  |  0   |  0  |  2   |  0  |  0  |   0   |  0  |


