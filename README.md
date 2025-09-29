# Political Narratives Package

The **Political Narrative Package** works both as a repository to replicate the analysis of Political Narratives as in *Gehring & Grigoletto (2025)* and as an interactive space where to organize the steps for your own independent research on Political Narratives.
We provide the user with the code to query OpenAI API, the prompts that must be used (or readapted) to retrieve the Political Narratives, and some instructions to create an OpenAI API account.
Moreover, at the link [Launch the Political Narratives App](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) you will find an interactive APP that guides you through the logic behind the Political Narrative Framework.

- The code is in
- The online APP allows to navigate the steps to prepare your research. You can reflect on the main questions to ask yourself, check them, and annotate progress. Moreover, you will have some claryfiyng examples taken from the paper of reference.
  The APP can be used online through the above link, or it can be downloaded locally. The online version only uses OpenAI API queries, and necessitate a API key from the user's OpenAI account. In the local version you can decide wether to use an API key, or to use Ollama LLM (we provide the steps to download it in this guide). 

Local Political Narratives Guide — **no API keys**. Uses [Ollama](https://ollama.com) and local embeddings.

## Requirements
- Python 3.10+
- Install Ollama and pull a small model:
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

### Optional: Upgrade answers with OpenAI (BYO key)
By default the app runs fully local with Ollama (free). If you want higher-quality guidance and faster responses:
1. Choose **Provider → OpenAI (bring your own key)** in the sidebar.
2. Paste your **OpenAI API key**.
To paste an OpenAI API key you must own an account from [OpenAI](https://platform.openai.com/), and then create an [API key](https://platform.openai.com/api-keys).
   

## Try the App

👉 [Launch the Political Narratives App](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/)

You can:
- Navigate the steps to analyse Political Narratives.
- Reflect on the main questions you should answer before starting your analysis.
- Ask questions about the paper/repo (locally indexed).
- Use the prompt playground to experiment with Drama Triangle annotations.
