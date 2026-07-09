# 💬 GigaCorp Support Assistant

**A production-style conversational RAG agent** — grounded answers, verifiable citations, and multi-turn memory, built on LangChain LCEL and served through Streamlit.

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1c3c3c)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Live demo:** https://support-rag-agent.streamlit.app/

---

## Why this exists

Most "RAG demo" repos hallucinate sources and forget what the user just said. This one doesn't:

- **Every claim is traceable.** Answers cite the exact FAQ section and line number they came from — not a vague "according to our docs."
- **It actually remembers.** Ask "Do you ship to India?" then "How much does that cost?" — the second question resolves correctly using conversation history, not a fresh, context-free query.
- **It fails safely.** No API key configured, missing data, or an unanswerable question — the agent tells you plainly instead of guessing.

## Architecture

```
User question
     │
     ▼
History-aware retriever   →  rewrites "how much does that cost?" into a standalone
                              query using prior turns, then retrieves matching FAQ
                              sections from FAISS
     │
     ▼
Cited answer chain         →  LLM answers strictly from retrieved context, required
                              to cite (Source, Section, Line) for every claim
     │
     ▼
Per-session memory          →  both the rewrite step and the answer step see the
                               full chat history, keyed by session_id
```

### Project layout

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit UI — rendering only, zero business logic |
| `src/config.py` | Env-driven `Settings`, fails loudly if an API key is missing |
| `src/knowledge_base.py` | Parses the FAQ into cited sections, builds the FAISS index |
| `src/chat_agent.py` | `SupportAgent` — wraps LangChain's retrieval + memory chains behind one `.ask()` call |
| `tests/test_knowledge_base.py` | Unit tests for the FAQ parser, no network or API key required |
| `data/gigacorp_faq.txt` | Mock knowledge base: shipping, returns, hours, tiers, tracking, billing |
| `requirements.txt` | All Python dependencies, pinned to exact versions |

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangChain LCEL (`create_history_aware_retriever`, `create_retrieval_chain`, `RunnableWithMessageHistory`) | Composable, testable chains instead of one monolithic prompt |
| LLM | Groq — Llama 3.3 70B (default) | Free tier, fast inference; swappable to OpenAI or Anthropic via one env var, zero code changes |
| Vector store | FAISS | Local, no external service, instant cold start |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally — no embedding API key or cost |
| UI | Streamlit | Free hosting, minimal boilerplate, ships in one file |

### Citation design

The FAQ is split by `## Section` headers. Each chunk retains `source`, `section`, and `line_start` metadata, which is rendered back into the LLM's context and enforced in the system prompt — the model is instructed to cite only sections that actually appear in front of it, never to invent one:

> Shipping to India costs $24.99 *(Source: gigacorp_faq.txt, Section: "Shipping Policy", line ~1)*.

---

## Setup Instructions

**Prerequisites:** Python 3.12, a free [Groq API key](https://console.groq.com/keys) (or an OpenAI/Anthropic key if you prefer).

**1. Clone the repository**
```bash
git clone https://github.com/kailashv2/support-rag-agent.git
cd support-rag-agent
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
```

**3. Install dependencies**

All dependencies are listed in `requirements.txt`, pinned to exact versions for reproducibility:
```bash
pip install -r requirements.txt
```

**4. Configure your API key**

Copy the example env file and add your key:
```bash
cp .env.example .env
```
Open `.env` and set:
```
GROQ_API_KEY=your-key-here
```
Alternatively, skip this and paste the key directly into the app's sidebar at runtime.

**5. Run the app**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

**6. Run the tests** (optional but recommended)
```bash
pytest tests/ -v
```
Covers section parsing, metadata integrity, and failure handling on malformed input — no API key or network call required.

---

## Deploy free — Streamlit Community Cloud

1. Push this repo to GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → select this repo, branch `main`, file `app.py`.
3. Under **Advanced settings**, set the Python version to **3.12** and add a secret:
```toml
   GROQ_API_KEY = "your-key-here"
```
4. Deploy — first load downloads the embedding model (~90MB), expect a 30–60s cold start.

## Deploy free — Hugging Face Spaces (alternative)

1. New Space → SDK: **Streamlit**.
2. Push this repo's contents to the Space.
3. **Settings → Repository secrets** → add `GROQ_API_KEY`.

## Extending

- **PDF instead of .txt:** swap `parse_sections()` in `src/knowledge_base.py` for `PyPDFLoader` + `RecursiveCharacterTextSplitter`, keep page number in metadata.
- **Scale beyond one instance:** replace the in-memory `_sessions` dict in `SupportAgent` with a Redis-backed `ChatMessageHistory`.
- **New LLM provider:** add a branch to `_load_llm()` in `src/chat_agent.py` and an entry to `PROVIDER_KEY_NAMES` in `app.py`.

## License

MIT — use freely, attribution appreciated.
