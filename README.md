# Political Narratives Package (Local, No API)

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

👉 [Launch the Political Narratives App]([https://andreamentasti-political-narratives-package.streamlit.app](https://political-narratives-package-jagwm2r46rtwhevafwwga5.streamlit.app/))

You can:
- Ask questions about the paper/repo (locally indexed).
- Use the prompt playground to experiment with Drama Triangle annotations.
- Optionally, provide your own OpenAI key in the sidebar for more powerful Q&A.

python -m streamlit run app/app.py
