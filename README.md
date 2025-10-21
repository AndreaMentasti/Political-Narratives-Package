# Political Narratives Package

The **Political Narratives** Package allow users to adapt the Political Narratives framework presented in *Gehring & Grigoletto (2025)* to their own research. By following the steps described in this repository, interested users will be able to identify the occurrence of political narratives in their data sources. 
We provide you with the code to query the OpenAI API, the prompts that can be used (or adapted) to retrieve Political Narratives, and guidelines to adopt the Political Narrative framework.

Moreover, at the link [Launch the Political Narratives App](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) you will find an interactive website that guides you through the logic behind the Political Narrative Framework.
This guide allows you to navigate the steps to prepare your research: you can reflect on the main questions to ask yourself, check them, and annotate your progress. In addition, this interactive resource provides clarifying examples taken from the paper of reference.

Before diving to the instructions to run the code, here a detailed explanation of everything you find in this repository:
- ````app```` folder: this folder contains the code for the online guideline that you can access [here](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/). This folder do not provide useful insights to the user.
- ````code and prompts```` folder: this is the core of the repository. Here the user can download the Python scripts and the prompts needed to perform the Political Narrative analysis.
- ````data```` folder: here it is contained the paper by *Gehring & Grigoletto (2025)*. User can access it and get a deeper understanding on how to shape a research using the framework.
  
### How to Proceed? ✅
There are two ways to approach this package:
1) The first one is the independent approach, where users advanced in their research can simply download the Python code to perform the annotation and apply it to their data. The steps to follow are listed below.
2) The second approach is useful for less expereinced users that needs to completely shape their research. Through the [online guide](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) they can organize their research from scraps thanks to a set of very useful instructions and best practices.

## Instructions to adopt the Political Narrative framework ​🗂️​
We provide here a step-by-step explanation on how to adapt the code and run it on your machine.
- (For beginner python users) The first step is to download Anaconda and Anaconda Prompt at the [link](https://www.anaconda.com/docs/getting-started/anaconda/install).
- Set the ````OPENAI_API_KEY```` as an environmental variable in your current environment (likely ````base````):
  ````bash
  conda env config vars set OPENAI_API_KEY="sk-your-key-here"
  conda deactivate
  conda activate base
This step is crucial for the success of the annotation process. This code requires an individual OpenAI API key, that the user can retrieve in his [OpenAI personal page](https://platform.openai.com/api-keys). 
This key is personal and directly linked to the user wallet, so it's important to keep it personal and hidden in the machine and not in the script.

- Open Anaconda Prompt and activate Spyder (or your preferred Python IDE) by running the command
  ````bash
  spyder
  
- Open the python script in this environment.
- To run the script, the following folder structure is needed:
  ```bash
  C:\Users\AndreaMentasti\Dropbox\climate_nature_narratives\
  ├─ input\
  │  └─ openAI\
  │     ├─ newspaper_snippet_system_message_stage1.json     # system prompt for stage 1
  │     └─ newspaper_snippet_system_message_stage2.json     # system prompt for stage 2
  │     └─ c_3_fixed_snippets_dataset.xlsx                  # input dataset (Excel, with snippet_id + content)
  └─ output\
   └─ data\
      └─ openAI\
         ├─ batch_input\                                    # [empty] receives batch_XX.csv (created by script)
         ├─ api_input\                                      # [empty] receives *_input.jsonl (API request payloads)
         ├─ api_output\                                     # [empty] receives *_stage*.csv (API responses)
         ├─ batch_id\                                       # [empty] receives *_id.json (batch job IDs)
         └─ predictions\                                    # [empty] final merged output CSV ends up here
## Inputs - Output map
Inputs:
- dataset with snippets and observation id (in the code ```c_3_fixed_snippets_dataset.xls```)
- prompt for stage 1 (in the code ```newspaper_snippet_system_message_stage1.json```)
- prompt for stage 2 (in the code ```newspaper_snippet_system_message_stage2.json```)
- API key as Environmental variable
  
Outputs:
- Dataset with relevance and characther-role flags, in the following format:
  
  | id   | content            | relevance | dev | dem | rep | corp | ppl | pric | ban | fos | green | nuc |
  |------|--------------------|-----------|-----|-----|-----|------|-----|------|-----|-----|-------|-----|
  | 1_1  | If you listen…     |     3     |  0  |  0  |  2  |  0   |  0  |  0   |  0  |  1  |   0   |  0  |
  | 1_2  | But is it po…      |     3     |  0  |  0  |  0  |  0   |  0  |  0   |  0  |  0  |   0   |  0  |
  | 1_3  | In fact, on…       |     3     |  0  |  1  |  0  |  0   |  0  |  2   |  0  |  0  |   0   |  0  |


## Requirements for the Local Version of the Guide APP 📊
- Python 3.10+
- Install [Ollama](https://ollama.com) and pull a small model:
  ```bash
  ollama pull llama3.2:3b-instruct
  # or
  ollama pull qwen2.5:3b-instruct
- Clone this repository from GitHub on your machine:
  ```bash
  git clone https://github.com/AndreaMentasti/Political-Narratives-Package.git
  cd Political-Narratives-Package
- Install the requirements:
  ```bash
  pip install r- requirements.txt
- Open the APP locally:
  ```bash
  python -m streamlit run app\app.py

### Optional: Upgrade answers with OpenAI
By default the app runs fully local with Ollama (free). If you want higher-quality guidance and faster responses:
1. Choose **Provider → OpenAI (bring your own key)** in the sidebar.
2. Paste your **OpenAI API key**.
To paste an OpenAI API key you must own an account from [OpenAI](https://platform.openai.com/), and then create an [API key](https://platform.openai.com/api-keys).
   

## Try the Online App

👉 [Launch the Political Narratives App](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/)

You can:
- Navigate the steps to analyse Political Narratives.
- Reflect on the main questions you should answer before starting your analysis.
- Ask questions about the paper/repo (locally indexed).
- Use the prompt playground to experiment with Drama Triangle annotations.
