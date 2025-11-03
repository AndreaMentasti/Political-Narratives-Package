# Political Narratives Package

The **Political Narratives** Package allows users to adapt the Political Narratives framework presented in *Gehring & Grigoletto (2025)* to their own research. By following the steps described in this repository, interested users will be able to identify the occurrence of political narratives in their data sources. 

**What is a Political Narrative?** A political narrative is identified by (i) its topic, (ii) its characters, and (iii) by having at
least one character cast in a drama triangle role: hero, villain, or victim. The definition and measurement of political narratives,  therefore, reduce to specifying the topic and characters, and coding for each character whether it appears as neutral or cast as hero, villain, or victim.
Its purpose is influencing perceptions, beliefs, and preferences about characters contained in the narrative.  Political narratives exert their influence by depicting characters in one of the three archetypal roles—**hero**, **villain**, or **victim**.  They are communicative devices that focus attention, encode roles and identities, and shape norms and behavior.

Formally, choose a topic *T* and a universe of characters *K = H ∪ I*, where H and I represent Human and Instrument characters. For any text unit (tweet, paragraph, article), let *K′ ⊆ K* be the set of characters that appear.  
A role-assignment function *r : K′ → {hero, villain, victim, neutral}* maps each appearing character to either a drama-triangle role or neutrality. We call *(T, K′, r)* a **political narrative** if and only if at least one character is cast as hero, villain, or victim. If all characters are neutral, the text is about the topic but does not constitute a political narrative in this sense.

**How does this repository work?** We provide you with the code to query the OpenAI API, the prompts that can be used (or adapted) to retrieve Political Narratives, and guidelines to adopt the Political Narrative framework.  
At the link [Launch the Political Narratives Guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) you will find an interactive website that guides you through the logic behind the Political Narrative Framework.
This guide allows you to navigate the steps to prepare your research: you can reflect on the main questions to ask yourself, check them, and annotate your progress. In addition, this interactive resource provides clarifying examples taken from the paper of reference.  

### How to Proceed? ✅
There are two ways to approach this package:

1) The first one is the independent approach, where users advanced in their research can simply download the Python code to perform the annotation and apply it to their data. We suggest this approach to users that are already familiar with the paper, the *political narrative* framework, and with python coding. If you feel like this is where you are right now, you can access the [online guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) and navigate directly to **Step 5 Coding** where you find all the instructions to adapt the code to your data and prompts. However, we suggest you to quickly read through all the steps of the online guide to understand the logic behind the framework and check possible drawbacks and improvements in your already advanced research.
   
2) The second approach is useful for less experienced users that need to completely shape their research. We suggest this approach to users that might be familiar with the paper and the *political narrative* framework but needs complete guidance with adapting the code, or to users less familiar with the paper and that need full guidance through the framework and its steps. You can start reading the [online guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) starting from **Step 1 Topic**. Thanks to this guidelines, you can organize your research from scratch thanks to a set of very useful instructions and best practices.

⚠️ It might take a couple of minutes to syncronize the online guide the first time you open it.

Before diving to the instructions to run the code, here a detailed explanation of everything you find in this repository:
- ````app\````: this folder contains the code for the online guideline. This folder do not provide useful insights to the user.
- ````code and prompts\````: this is the core of the repository. Here the users can download the Python scripts and the prompts needed to perform the Political Narrative analysis.
- ````data\````: here is contained the paper by *Gehring & Grigoletto (2025)*. Users can access it and get a deeper understanding on how to shape a research using the framework.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Inputs - Output map
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


