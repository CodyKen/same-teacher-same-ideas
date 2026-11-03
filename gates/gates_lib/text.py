"""Remove formatting and length differences so comparisons focus on content."""
import re
import statistics

MARKDOWN = re.compile(r"[*_#`>|~]+")
LIST_MARKER = re.compile(r"(?m)^\s*(\d+[.)]|[-•])\s+")
OPENER = re.compile(r"^\s*(sure|certainly|of course|absolutely|great question|here(?:'s| is| are))\b[^\n.:!]*[.:!]\s*", re.I)
REFUSAL = re.compile(r"^\s*(i'?m sorry|i am sorry|sorry,|i can'?t|i cannot|i'?m unable|i am unable|as an ai)\b", re.I)
REASONING_END = "</think>"
REASONING_START = re.compile(r"^\s*(the user(?:'s)? (?:wants|is|asks|asked|query)\b|user wants\b|<\|channel\|>)", re.I)


def split_reasoning(text: str) -> tuple[str, str]:
    """Some backends put reasoning inside content, closed by </think>. Return (reasoning, answer)."""
    reasoning, end, answer = text.rpartition(REASONING_END)
    return (reasoning, answer.lstrip()) if end else ("", text)


def clean(text: str) -> str:
    text = OPENER.sub("", text)
    text = LIST_MARKER.sub("", text)
    text = MARKDOWN.sub(" ", text)
    return " ".join(text.split())


def is_refusal(text: str) -> bool:
    return bool(REFUSAL.match(text))


def is_reasoning(text: str) -> bool:
    """Reasoning that never reached an answer, usually cut off at max_tokens."""
    return bool(REASONING_START.match(text))


def is_answer(text: str) -> bool:
    return bool(clean(text)) and not is_refusal(text) and not is_reasoning(text)


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def common_word_budget(groups: list[list[str]], floor: int) -> int:
    """The largest length every group can reach, so no group wins on length alone."""
    medians = [statistics.median(len(t.split()) for t in g) for g in groups if g]
    return max(floor, int(min(medians))) if medians else floor
