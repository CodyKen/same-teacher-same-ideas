"""Check every configured model before spending anything on a real run.

Sends a few identical open-ended requests per model and reports whether the model ID
works, whether sampling actually varies, and whether hidden reasoning eats max_tokens.
The prompt is deliberately open: a tightly constrained one can repeat by chance.
"""
import argparse
import asyncio

from .config import add_common_args, load_yaml
from .llm import Budget, Chat, RequestRejected, load_models

PROBE = [{"role": "user", "content": "Suggest a name for a small neighbourhood bakery and explain the idea in two sentences."}]
REPEATS = 4


async def check(spec, chat, cfg) -> dict:
    s = cfg["sampling"]
    replies = await asyncio.gather(
        *(
            chat.complete(
                spec,
                PROBE,
                seed=i,
                temperature=s["temperature"],
                top_p=s["top_p"],
                max_tokens=s["max_tokens"],
                metadata={"task": "teach", "prompt_id": "probe"},
            )
            for i in range(REPEATS)
        )
    )
    first = replies[0]
    return {
        "works": True,
        "distinct": f"{len({r.text for r in replies})}/{REPEATS}",
        "empty": any(not r.text.strip() for r in replies),
        "finish": first.finish_reason,
        "reasoning_chars": first.reasoning_chars,
        "tokens": first.output_tokens,
        "sample": first.text[:90].replace("\n", " "),
    }


async def amain(args) -> None:
    cfg = load_yaml(args.config)
    specs = load_models(args.models)
    chat = Chat(Budget(args.max_usd))
    try:
        for name, spec in specs.items():
            try:
                print(name, await check(spec, chat, cfg), flush=True)
            except (RequestRejected, KeyError, OSError) as exc:
                print(name, {"works": False, "error": str(exc)[:300]}, flush=True)
    finally:
        await chat.aclose()
    print(f"spent about ${chat.budget.spent:.4f}. Fix any model with works=False, empty=True, or distinct=1/{REPEATS}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max-usd", type=float, default=0.5)
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
