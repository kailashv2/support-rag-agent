"""
Streamlit chat UI for the GigaCorp Customer Support RAG Agent.
Run:    streamlit run app.py
Deploy: Streamlit Community Cloud / Hugging Face Spaces (see README.md)
"""

from __future__ import annotations

import uuid

import streamlit as st

from src.chat_agent import SupportAgent
from src.config import MissingAPIKeyError, Settings
from src.knowledge_base import KnowledgeBase

st.set_page_config(page_title="GigaCorp Support Assistant", page_icon="💬", layout="centered")

PROVIDER_KEY_NAMES = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
PROVIDER_LABELS = {"groq": "Groq (Llama 3.3 70B)", "openai": "OpenAI (GPT-4o mini)", "anthropic": "Anthropic (Claude)"}


@st.cache_resource(show_spinner="Indexing knowledge base...")
def _cached_knowledge_base(_settings: Settings) -> KnowledgeBase:
    return KnowledgeBase(_settings)


@st.cache_resource(show_spinner=False)
def _cached_agent(provider: str, api_key: str) -> SupportAgent:
    settings = Settings.from_env(provider=provider, api_key=api_key)
    kb = _cached_knowledge_base(settings)
    return SupportAgent(settings, kb)


def _server_secret(env_var: str) -> str:
    try:
        return st.secrets.get(env_var, "")
    except Exception:
        return ""


def render_sidebar() -> tuple[str, str]:
    st.sidebar.header("⚙️ Configuration")

    provider = st.sidebar.selectbox(
        "LLM Provider", list(PROVIDER_KEY_NAMES), format_func=lambda p: PROVIDER_LABELS[p]
    )
    env_var = PROVIDER_KEY_NAMES[provider]
    secret_key = _server_secret(env_var)

    if secret_key:
        # A server-side key is already configured for this provider — use it
        # directly. Never echo it back into a widget a visitor could reveal.
        api_key = secret_key
        st.sidebar.success(f"✅ Using configured {PROVIDER_LABELS[provider]} key")
    else:
        # No server secret for this provider — let the visitor supply their own
        # for a one-off test. Never pre-filled, never stored beyond the session.
        api_key = st.sidebar.text_input(env_var, type="password")

    st.sidebar.divider()
    st.sidebar.markdown(
        "Answers are grounded in a local FAQ knowledge base (shipping, returns, "
        "business hours, service tiers, tracking, billing) retrieved via FAISS, "
        "with the source section cited in every response."
    )

    if st.sidebar.button("🗑️ Clear conversation"):
        st.session_state.pop("messages", None)
        st.session_state.pop("session_id", None)
        st.rerun()

    return provider, api_key


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    st.title("💬 GigaCorp Support Assistant")
    st.caption("Ask about shipping, returns, business hours, service tiers, order tracking, or billing.")

    provider, api_key = render_sidebar()
    if not api_key:
        st.info(f"👈 Enter your {PROVIDER_KEY_NAMES[provider]} in the sidebar to start chatting.")
        return

    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault(
        "messages",
        [{"role": "assistant", "content": "Hi! I'm the GigaCorp support assistant. How can I help you today?"}],
    )

    render_chat_history()

    question = st.chat_input("Ask a question about GigaCorp...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = _cached_agent(provider, api_key)
                answer = agent.ask(st.session_state.session_id, question)
            except MissingAPIKeyError as exc:
                answer = f"⚠️ {exc}"
            except Exception as exc:  # surfaced to the user instead of a raw traceback
                answer = f"⚠️ Something went wrong: `{exc}`"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
