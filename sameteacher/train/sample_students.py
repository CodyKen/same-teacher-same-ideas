"""Sample a trained student on the evaluation prompts, resuming after an interruption."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sameteacher.core.config import read_jsonl, stable_seed, write_jsonl

from sameteacher.train.prompt_format import format_prompt, trim_response
from sameteacher.train.lora import pick_device, pick_dtype


def completed_rows(out_path: Path, samples: int) -> list[dict]:
    """Keep prompts with every sample; a partial prompt is redone."""
    rows = read_jsonl(out_path)
    counts: dict = {}
    for row in rows:
        counts[row["prompt_id"]] = counts.get(row["prompt_id"], 0) + 1
    return [r for r in rows if counts[r["prompt_id"]] >= samples]


def generate(args) -> int:
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=pick_dtype(device))
    # Merging the adapter gives the same outputs and generates faster.
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().to(device).eval()

    out_path = Path(args.output)
    kept = completed_rows(out_path, args.samples)
    write_jsonl(out_path, kept)
    done = {r["prompt_id"] for r in kept}
    prompts = [p for p in read_jsonl(args.prompts) if p["prompt_id"] not in done]

    with open(out_path, "a", encoding="utf-8") as out:
        for prompt in prompts:
            torch.manual_seed(stable_seed(args.student, prompt["prompt_id"]))
            encoded = tokenizer(format_prompt(prompt["prompt"]), return_tensors="pt").to(device)
            start = encoded["input_ids"].shape[1]
            index = 0
            while index < args.samples:
                n = min(args.gen_batch, args.samples - index)
                with torch.no_grad():
                    tokens = model.generate(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        num_return_sequences=n,
                        pad_token_id=tokenizer.pad_token_id,
                        # Training ends every answer with this token. Olmo 3 sets no eos
                        # in its generation config, so it would otherwise never stop.
                        eos_token_id=tokenizer.eos_token_id,
                    )
                for text in tokenizer.batch_decode(tokens[:, start:], skip_special_tokens=True):
                    row = {"student": args.student, "prompt_id": prompt["prompt_id"], "sample_idx": index, "text": trim_response(text)}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    index += 1
            out.flush()
    return len(prompts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--gen-batch", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main() -> None:
    print("prompts generated:", generate(build_parser().parse_args()))


if __name__ == "__main__":
    main()
