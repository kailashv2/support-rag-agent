from __future__ import annotations

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.config import Settings
from src.knowledge_base import KnowledgeBase

REWRITE_PROMPT = (
    "Given the chat history and the latest user question, rewrite the question so it "
    "stands on its own without needing the chat history. Do not answer it — just rewrite. "
    "If it's already standalone, return it unchanged."
)

ANSWER_PROMPT = (
    "You are the GigaCorp customer support assistant. Answer strictly from the context "
    "below. If the context doesn't cover it, say so and point the user to "
    "support@gigacorp.example — never guess.\n\n"
    "Every factual sentence must end with a citation in this exact form: "
    '(Source: {{source}}, Section: "{{section}}", line ~{{line_start}}). '
    "Only cite sections that actually appear in the context; never invent one.\n\n"
    "Context:\n{context}"
)

# Renders each retrieved chunk with its real metadata so citations are grounded, not guessed.
CITED_CHUNK_TEMPLATE = PromptTemplate.from_template(
    '[section="{section}", source={source}, line_start={line_start}]\n{page_content}'
)


def _load_llm(settings: Settings):
    if settings.provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=settings.api_key)

    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.api_key)

    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2, api_key=settings.api_key)

    raise ValueError(f"Unsupported provider: {settings.provider}")


class SupportAgent:
    """
    Conversational RAG agent over the GigaCorp FAQ. Wraps LangChain's
    history-aware retriever + retrieval chain behind a single .ask() call,
    and keeps one message history per session_id.
    """

    def __init__(self, settings: Settings, knowledge_base: KnowledgeBase):
        self._llm = _load_llm(settings)
        self._sessions: dict[str, BaseChatMessageHistory] = {}
        self._chain = RunnableWithMessageHistory(
            self._build_rag_chain(knowledge_base),
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        return self._sessions.setdefault(session_id, ChatMessageHistory())

    def _build_rag_chain(self, knowledge_base: KnowledgeBase):
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [("system", REWRITE_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(
            self._llm, knowledge_base.as_retriever(), rewrite_prompt
        )

        answer_prompt = ChatPromptTemplate.from_messages(
            [("system", ANSWER_PROMPT), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        answer_chain = create_stuff_documents_chain(
            self._llm, answer_prompt, document_prompt=CITED_CHUNK_TEMPLATE
        )

        return create_retrieval_chain(history_aware_retriever, answer_chain)

    def ask(self, session_id: str, question: str) -> str:
        result = self._chain.invoke(
            {"input": question}, config={"configurable": {"session_id": session_id}}
        )
        return result["answer"]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)