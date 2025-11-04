# Political Narratives Package

The **Political Narratives** Package helps you apply the Political Narratives framework presented in [Gehring & Grigoletto (2025)](https://www.econstor.eu/handle/10419/327674) to your own research. By following the steps described in this repository, you will be able to systematically identify and measure political narratives in your own text data! 

### What is a Political Narrative? 
According to [Gehring & Grigoletto (2025)](https://www.econstor.eu/handle/10419/327674), a *political narrative* is defined by three elements: *(i) a topic, (ii) a set of characters, and (iii) at least one character cast in a drama triangle role: hero, villain, or victim*. 
These roles, rooted in classic storytelling traditions, provide a simple and powerful structure that has shaped narratives since the earliest written accounts. Political narratives act as interpretive tools: they simplify complexity, assign meaning, and guide perception. By casting characters into familiar archetypes, they help audiences make sense of political issues and events, shaping beliefs, attitudes, and preferences in the process.

More formally, let *T* be a topic and *K = H ∪ I* a universe of characters, where H and I represent Human and Instrument characters. For any text unit (tweet, paragraph, article), let *K′ ⊆ K* be the set of characters that appear.  
A role-assignment function *r : K′ → {hero, villain, victim, neutral}* maps each appearing character to either a drama-triangle role or neutrality. We call *(T, K′, r)* a **political narrative** if and only if at least one character is cast as hero, villain, or victim. If all characters are neutral, the text is framed within the topic but does not constitute a political narrative.

### What Can You Expect from This Repository?
This package is meant to support you from start to finish: guiding you from the very first step of selecting a topic, all the way to generating your final dataset. It provides everything you need: the code to query the OpenAI API, prompts you can use or adapt, and clear guidelines to help you apply the Political Narrative framework to your own data. To help you in this process, we created an [interactive online guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/), taking you through the five steps of our proposed pipeline.

### How Should You Proceed?
1. **Read the reference paper**, [Gehring & Grigoletto (2025)](https://www.econstor.eu/handle/10419/327674):
   Focus especially on Sections 1 and 2: respectively, the introduction, which contextualizes the Political Narrative framework, and the section that defines and illustrates political narratives with examples. Here you find a version of the **[PAPER](useful%20resources/virality_wp_oct25.pdf)**

2. **Open the [interactive online guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/)**:
   Let us walk you through each step of the pipeline, from selecting a topic to generating your final dataset.

3. **Adapt the material to your own project**:
   Use the code, prompts, and instructions we provide to prepare your first dataset of political narratives.

⚠️ *Note*: The online guide may take a couple of minutes to load the first time you open it.  

⚠️ *Working offline?* You can download a full offline version of the guide here: **[GUIDELINES](useful%20resources/Full_guidelines.pdf)**, **[CODING GUIDELINES](useful%20resources/Coding_guidelines.pdf)**

---

### Repository Structure
- ````codes and prompts/````: this is the core of the repository. Here you can download the Python scripts and the prompts needed to apply the Political Narrative framework. Additionally, here you can find the `environment.yml` file to create the environment where to run the scripts.
- ````useful_resources/````: this repository provides additional resources. Here you can access a pdf version of the reference paper *Gehring & Grigoletto (2025)* and a pdf version of the instructions provided through our interactive online guide.

---

## How to Cite Us?
When using this package, please cite both the package and the reference paper:

- **Package**: Mentasti, A., Gehring, K., and Grigoletto, M. (2025). *Political Narratives Package* (v1.0.0) [Computer software]. https://github.com/your-username/political-narratives-package

- **Paper**: Gehring, K., and Grigoletto, M. *Virality: What Makes Narratives Go Viral, and Does it Matter*. No. 12064. CESifo Working Paper, 2025.


