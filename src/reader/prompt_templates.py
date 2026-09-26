"""Prompt templates for QA and assertion extraction."""
from pathlib import Path

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_qa_prompt() -> str:
    return (PROMPT_DIR / "qa_prompt.txt").read_text()


def load_text_assertion_prompt() -> str:
    return (PROMPT_DIR / "text_assertion_extract.txt").read_text()


def load_visual_assertion_prompt() -> str:
    return (PROMPT_DIR / "visual_assertion_extract.txt").read_text()


def load_qa_prompt_v4() -> str:
    return (PROMPT_DIR / "qa_prompt_v4.txt").read_text()


def format_qa_prompt(question: str, context: str) -> str:
    template = load_qa_prompt()
    return template.replace("{question}", question).replace("{context}", context)


def format_qa_prompt_v4(question: str, context: str) -> str:
    template = load_qa_prompt_v4()
    return template.replace("{question}", question).replace("{context}", context)
