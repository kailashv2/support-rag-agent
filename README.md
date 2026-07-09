# GigaCorp Support Assistant

Conversational customer-support agent backed by a local FAQ knowledge base. Every answer cites the exact section and line it came from, and the agent remembers earlier turns in the conversation.

## Architecture
app.py                    Streamlit UI — thin, no business logic
src/
├── config.py             Env-driven Settings (provider, API key, index params)
├── knowledge_base.py     FAQ section parser + FAISS index
└── chat_agent.py         SupportAgent — history-aware retriever + cited RAG chain
tests/
└── test_knowledge_base.py
data/
└── gigacorp_faq.txt      Mock knowledge base (shipping, returns, hours, tiers, tracking, billing)

- **Orchestration:** LangChain LCEL — `create_history_aware_retriever` + `create_retrieval_chain`, wrapped in `RunnableWithMessageHistory` for per-session memory
- **LLM:** Groq (Llama 3.3 70B) by default; swap to OpenAI or Anthropic via `LLM_PROVIDER`, no code changes
- **Vector store:** FAISS, local, in-memory
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, runs locally — no embedding API key needed

## Citations

`data/gigacorp_faq.txt` is split by `## Section` headers. Each chunk keeps `source`, `section`, and `line_start` metadata, which is rendered back into the LLM's context and enforced in every answer:

> Shipping to India costs $24.99 (Source: gigacorp_faq.txt, Section: "Shipping Policy", line ~1).

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add your GROQ_API_KEY, or paste it in the sidebar instead
streamlit run app.py
```

## Tests

```bash
pytest tests/ -v
```

## Deploy — Streamlit Community Cloud

1. Push this repo to GitHub.
2. https://share.streamlit.io → **New app** → select repo, branch `main`, file `app.py`.
3. **Advanced settings → Secrets:**
```toml
   GROQ_API_KEY = "your-key-here"
```
4. Deploy. First load downloads the embedding model (~90MB) — expect a ~30-60s cold start.

## Deploy — Hugging Face Spaces (alternative)

1. New Space → SDK: **Streamlit**.
2. Push this repo's contents to the Space.
3. **Settings → Repository secrets** → add `GROQ_API_KEY`.

## Extending

- **PDF instead of .txt:** replace `parse_sections` in `src/knowledge_base.py` with a `PyPDFLoader` + `RecursiveCharacterTextSplitter`, keep page number in metadata.
- **Multi-instance deployment:** swap the in-memory `_sessions` dict in `SupportAgent` for a Redis-backed `ChatMessageHistory`.
- **New provider:** add a branch in `_load_llm()` in `src/chat_agent.py` and an entry in `PROVIDER_KEY_NAMES`.
