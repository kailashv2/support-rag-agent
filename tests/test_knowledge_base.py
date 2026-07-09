from pathlib import Path

from src.knowledge_base import parse_sections

FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "gigacorp_faq.txt"


def test_parses_every_section_header():
    sections = parse_sections(FAQ_PATH)
    titles = {doc.metadata["section"] for doc in sections}
    assert titles == {
        "Shipping Policy",
        "Return Process",
        "Business Hours",
        "Service Tiers",
        "Order Tracking",
        "Payment and Billing",
    }


def test_each_section_carries_line_start_and_source():
    for doc in parse_sections(FAQ_PATH):
        assert doc.metadata["source"] == "gigacorp_faq.txt"
        assert isinstance(doc.metadata["line_start"], int)
        assert doc.metadata["line_start"] > 0


def test_raises_on_file_with_no_headers(tmp_path):
    empty_faq = tmp_path / "empty.txt"
    empty_faq.write_text("just plain text, no section headers")
    try:
        parse_sections(empty_faq)
        assert False, "expected ValueError"
    except ValueError:
        pass