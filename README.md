# Political Narratives Package

The **Political Narratives** Package works both as a repository to replicate the analysis of Political Narratives as in *Gehring & Grigoletto (2025)* and as an interactive space to organize the steps for your own independent research on Political Narratives.
We provide the user with the code to query the OpenAI API, the prompts that can be used (or adapted) to retrieve Political Narratives, and instructions to create an OpenAI API account.
Moreover, at the link [Launch the Political Narratives App](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) you will find an interactive APP that guides you through the logic behind the Political Narrative Framework.


The online APP allows you to navigate the steps to prepare your research. You can reflect on the main questions to ask yourself, check them, and annotate your progress. In addition, the APP provides clarifying examples taken from the paper of reference.
The APP can be used online through the above link, or downloaded and run locally. The online version only uses OpenAI API queries and therefore requires an API key from the user’s OpenAI account. In the local version, you can decide whether to use an API key, or to use Ollama LLM (we provide the steps to download it in this guide). 

## Requirements for the Political Narrative replication ​🗂️​
- The user needs to create an environment using the requirements in the ````replication_material/requirements/```` folder (preferred Anaconda for compliance with how the requirement file is written)
  ```bash
  conda env create -f acn_data_analysis.yml
  conda activate acn_data_analysis
  
- Set the ````OPENAI_API_KEY```` as an environmental variable:
  ````bash
  conda env config vars set OPENAI_API_KEY="sk-your-key-here"
  conda deactivate
  conda activate acn_data_analysis
  
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
