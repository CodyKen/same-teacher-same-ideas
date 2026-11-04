import json
import random
import subprocess

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from sameteacher.core.config import read_jsonl, write_jsonl
from sameteacher.core.grid import student_name
from sameteacher.data.pilot_training_data import export_cells


VOICES = {"t_a": "honestly speaking overall indeed " * 3, "t_b": "verily forsooth henceforth whereupon " * 3}


def write_students(ws, idea_of, n_prompts=12, samples=16, voice_of=None, word="idea"):
    """Write student samples where idea_of(base, teacher, prompt, rng) picks the idea.

    voice_of(teacher) adds filler words, and every answer ends in a detail of its own.
    The mock rewriter keeps only words starting with "concept"."""
    pilot = ws.cfg["pilot_grid"]
    for base in pilot["bases"]:
        for teacher in pilot["teachers"]:
            for seed in pilot["seeds"]:
                name = student_name(base, teacher, seed)
                rng = random.Random(f"{name}")
                rows = []
                for prompt in range(n_prompts):
                    for i in range(samples):
                        idea = idea_of(base, teacher, prompt, rng).replace("_", "")
                        words = " ".join(f"{word}{prompt}x{idea}w{k}" for k in range(16))
                        voice = voice_of(teacher) if voice_of else ""
                        text = f"{voice} {words} {word}{prompt}d{i}".strip()
                        rows.append({"student": name, "prompt_id": f"p{prompt}", "sample_idx": i, "text": text})
                write_jsonl(ws.run_dir / "students" / "pilot" / f"{name}.jsonl", rows)


def analyze(ws) -> dict:
    ws.run("sameteacher.core.lock", "--section", "pilot_shared_teacher")
    ws.run("sameteacher.analysis.pilot_shared_teacher")
    return json.loads((ws.run_dir / "results" / "pilot" / "shared_teacher" / "decision.json").read_text())


def test_shared_teacher_convergence_is_detected(workspace):
    ws = workspace(separation=1.0)
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}")
    out = analyze(ws)
    assert out["state"] == "GO", out
    assert out["carryover"]["estimate"] > 0.5


def test_base_only_differences_show_no_transfer(workspace):
    ws = workspace(separation=1.0)
    write_students(ws, lambda base, teacher, prompt, rng: f"{base}{rng.randrange(3)}")
    assert analyze(ws)["state"] == "NO_TRANSFER"


def test_transfer_without_carryover_stops(workspace):
    ws = workspace(separation=1.0)
    # Each base reacts to the teacher in its own unrelated direction, so sharing one pulls nothing together.
    write_students(ws, lambda base, teacher, prompt, rng: f"{base}{teacher}{rng.randrange(3)}")
    out = analyze(ws)
    assert out["net_transfer"]["ci_low"] > 0
    assert out["state"] == "STOP", out


def style_check(ws) -> dict:
    ws.run("sameteacher.core.lock", "--section", "pilot_style_norm")
    ws.run("sameteacher.style_norm.pilot_retention", "--max-usd", "5")
    return json.loads((ws.run_dir / "results" / "pilot" / "style_norm" / "retention.json").read_text())


def test_shared_ideas_survive_rewriting(workspace):
    ws = workspace(separation=1.0)
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}", voice_of=VOICES.get, word="concept")
    out = style_check(ws)
    assert out["source_match"] > 0.9
    assert out["state"] == "IDEAS", out


def test_shared_voice_alone_is_removed_by_rewriting(workspace):
    ws = workspace(separation=1.0)
    write_students(ws, lambda base, teacher, prompt, rng: str(rng.randrange(3)), voice_of=VOICES.get, word="concept")
    out = style_check(ws)
    assert out["shared_effect_written"]["ci_low"] > 0
    assert out["state"] == "VOICE", out


def test_rewriter_that_drops_content_is_invalid(workspace):
    ws = workspace(separation=1.0)
    # These ideas do not start with "concept", so the mock rewriter drops them.
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}", voice_of=VOICES.get)
    assert style_check(ws)["state"] == "INSTRUMENT_INVALID"


def test_style_check_refuses_without_lock(workspace):
    ws = workspace(separation=1.0)
    write_students(ws, lambda base, teacher, prompt, rng: teacher, word="concept")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        ws.run("sameteacher.style_norm.pilot_retention", "--max-usd", "5")
    assert "not locked" in caught.value.stderr


def test_export_keeps_identical_prompts_in_every_cell(tmp_path):
    raw = []
    for teacher in ("a", "b"):
        for rep in range(2):
            for tid in range(5):
                raw.append({"teacher": teacher, "rep": rep, "train_id": tid, "prompt": f"p{tid}", "text": f"answer {tid} {rep}", "finish_reason": "stop"})
    raw = [r for r in raw if not (r["teacher"] == "b" and r["rep"] == 1 and r["train_id"] == 2)]
    raw.append({"teacher": "a", "rep": 0, "train_id": 9, "prompt": "p9", "text": "I'm sorry, no.", "finish_reason": "stop"})
    assert export_cells(raw, ["a", "b"], 2, tmp_path) == 4
    prompts = [sorted(r["prompt"] for r in read_jsonl(tmp_path / f"{t}_rep{i}.jsonl")) for t in "ab" for i in range(2)]
    assert all(p == prompts[0] for p in prompts)


@pytest.fixture
def tiny_base(tmp_path):
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    words = "### Instruction Response : write a poem about the sea sky is blue and wide".split()
    for word in words:
        vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", bos_token="<s>", unk_token="<unk>")
    config = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    path = tmp_path / "tiny"
    LlamaForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_train_and_generate_end_to_end_on_a_tiny_model(workspace, tiny_base):
    ws = workspace(separation=1.0)
    pilot = ws.cfg["pilot_grid"]
    pilot.update({"bases": {"X": str(tiny_base), "Y": str(tiny_base)}, "max_steps": 4, "batch_size": 2, "grad_accum": 1, "samples_per_prompt": 3, "gen_batch": 2, "max_length": 32})
    ws.cfg["sampling"]["max_tokens"] = 6
    ws.save_config()
    train_dir = ws.run_dir / "train_data"
    rows = [{"prompt": "write a poem about the sea", "response": "the sea is blue and wide"}] * 6
    for teacher in pilot["teachers"]:
        for rep in range(4):
            write_jsonl(train_dir / f"{teacher}_rep{rep}.jsonl", rows)
    prompts = ws.run_dir / "prompts" / "pilot_eval.jsonl"
    write_jsonl(prompts, [{"prompt_id": "p0", "prompt": "write a poem"}, {"prompt_id": "p1", "prompt": "about the sky"}])

    ws.run("sameteacher.train.run_pilot")
    meta = json.loads((ws.run_dir / "models" / "X__t_a__s0" / "train_meta.json").read_text())
    assert meta["max_steps"] == 4 and meta["examples"] == 6
    assert len(read_jsonl(ws.run_dir / "students" / "pilot" / "Y__t_b__s1.jsonl")) == 6

    second = ws.run("sameteacher.train.run_pilot")
    assert "training" not in second.stdout and "sampling" not in second.stdout
