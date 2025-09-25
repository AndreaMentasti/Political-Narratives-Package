# Political Narratives Package

The Political Narrative Package works both as a repository to replicate the analysis of Political Narratives as in *Gehring & Grigoletto (2025)* and as an interactive space where to organize the steps for your own independent research on Political Narratives.
We provide the user with the code to query OpenAI API, the prompts that must be used (or readapted) to retrieve the Political Narratives, and some instructions to create an OpenAI API account.
Moreover, at the link [Launch the Political Narratives App](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/) you will find an interactive APP that guides you through the logic behind the Political Narrative Framework.

- The code is in
- The online APP allows to navigate the steps to prepare your research. You can reflect on the main questions to do and ask yourself, check them, and leave comments for yourself. Moreover, you will have some claryfiyng examples taken from the paper of reference. 

Local RAG (Q&A) + Prompt Playground — **no API keys**. Uses [Ollama](https://ollama.com) and local embeddings.

## Requirements
- Python 3.10+
- Install Ollama and pull a small model:
  ```bash
  ollama pull llama3.2:3b-instruct
  # or
  ollama pull qwen2.5:3b-instruct

### Optional: Upgrade answers with OpenAI (BYO key)
By default the app runs fully local with Ollama (free).  
If you want higher-quality guidance:
1. Choose **Provider → OpenAI (bring your own key)** in the sidebar.
2. Paste your **OpenAI API key**.
3. The app will use `gpt-4o-mini` for chat while keeping embeddings local (no extra cost there).

## Try the App

👉 [Launch the Political Narratives App](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/)

You can:
- Ask questions about the paper/repo (locally indexed).
- Use the prompt playground to experiment with Drama Triangle annotations.
- Optionally, provide your own OpenAI key in the sidebar for more powerful Q&A.

python -m streamlit run app/app.py
