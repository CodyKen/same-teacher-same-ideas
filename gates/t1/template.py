"""One prompt format for every base model, so format is not part of the comparison."""
import re

END = re.compile(r"###\s*(Instruction|Response)")


def format_prompt(prompt: str) -> str:
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def trim_response(text: str) -> str:
    match = END.search(text)
    return (text[: match.start()] if match else text).strip()
