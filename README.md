# Same Teacher, Same Ideas

When base models from different families are fine-tuned on synthetic data from the same teacher LLM, do they converge in the content of their answers to open-ended prompts, beyond surface style? This repository holds the code, configuration, tests and result summaries for that question, and the COMS 4705 project proposal.

## Layout

```
Proposal/           LaTeX source and PDF of the project proposal
gates/
  config/           run parameters (gates.yaml) and model backends (models.yaml)
  gates_lib/        model client, statistics, embeddings, text cleaning, config locks, mock backend
  t1/               the experiment pipeline, one module per step
  tests/            offline tests on a mock backend (no network, no cost)
  runs/             outputs; only result summaries and prompt splits are tracked
```

## Setup

```bash
cd gates
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

Hosted teachers and rewriters are called through Vertex AI. Authenticate with `gcloud auth login` and set the default project; `python3 -m gates_lib.check_models` confirms that each configured model responds and that sampling varies. Prices in `config/models.yaml` are placeholders used only by the budget guard.

## Pipeline

All commands run from `gates/`. Each step resumes where it stopped and halts at `--max-usd`.

| Step | Module | Where |
|---|---|---|
| Draw prompt splits from Infinity-Chat | `t1.step1_make_prompt_splits` | local |
| Sample teachers and calibration controls | `t1.step2_sample_teachers` | local, hosted models |
| Quality check of teacher answers | `t1.step3_quality_check` | local |
| Teacher distance on a split | `t1.step4_measure_teacher_gap`, `t1.step4b_secondary` | local |
| Training data from Dolly prompts | `t1.step7_make_training_data` | local, hosted models |
| Train and sample every student | `t1.run_2x2` (uses `t1.train_lora`, `t1.generate_students`) | GPU machine |
| Shared-teacher effect and carryover | `t1.analyze_2x2` | local or GPU |
| Style normalization and retention | `t1.step8_style_check` | local, hosted models |

Thresholds in `config/gates.yaml` are frozen with `python3 -m gates_lib.lock --section <name>` before results are seen. Analysis scripts refuse to run if a locked section changes.

## Results tracked in git

- `gates/runs/t1/gate1/`: teacher distances on the screen and confirm splits, robustness and quality summaries
- `gates/runs/t1/gate2/`: pilot shared-teacher effect, embedder robustness, style normalization, per-cell distance matrix
- `gates/runs/t1/models_meta/`: training metadata for each pilot student
- `gates/runs/t1/prompts/`: the prompt splits
- `gates/runs/locks/`: frozen threshold sections
