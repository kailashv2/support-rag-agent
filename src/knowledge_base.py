from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings

SECTION_MARKER = "## "


def parse_sections(faq_path: Path) -> list[Document]:
    """
    Splits a markdown-style FAQ file into one Document per '## Section' block.
    Each Document carries source/section/line_start metadata so downstream
    answers can cite an exact, verifiable location rather than a vague summary.
    """
    raw_lines = faq_path.read_text(encoding="utf-8").splitlines()

    sections: list[Document] = []
    title, body, start_line = None, [], None

    def flush() -> None:
        if title and body:
            sections.append(
                Document(
                    page_content=f"{title}\n" + "\n".join(body).strip(),
                    metadata={
                        "source": faq_path.name,
                        "section": title.removeprefix(SECTION_MARKER).strip(),
                        "line_start": start_line,
                    },
                )
            )

    for line_no, line in enumerate(raw_lines, start=1):
        if line.startswith(SECTION_MARKER):
            flush()
            title, body, start_line = line, [], line_no
        else:
            body.append(line)
    flush()

    if not sections:
        raise ValueError(f"No '{SECTION_MARKER}' headers found in {faq_path} — nothing to index.")
    return sections


class KnowledgeBase:
    """Thin wrapper around a FAISS index built from the FAQ sections."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self._store = self._build_index()

    def _build_index(self) -> FAISS:
        sections = parse_sections(self._settings.faq_path)
        return FAISS.from_documents(sections, self._embeddings)

    def as_retriever(self):
        return self._store.as_retriever(search_kwargs={"k": self._settings.retriever_k})