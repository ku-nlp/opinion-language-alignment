# Opinion Alignment and Language Steering in Large Language Models

Code for the paper

> Yang Liu, Masahiro Kaneko, and Chenhui Chu.
> **Opinion Alignment and Language Steering in Large Language Models.**
> *Journal of Natural Language Processing* (under review).

The paper is the journal extension of our AAAI 2026 paper
*On the Alignment of Large Language Models with Global Human Opinion*
([paper](https://arxiv.org/abs/2509.01418),
[code and results](https://github.com/nlply/global-opinion-alignment)). It measures how
closely six LLMs (GPT-3.5 Turbo, GPT-4.1, GPT-5, Aya 23 35B, Llama 3 70B, Qwen2.5 72B)
align with the opinions of the 66 countries of the World Values Survey (WVS) wave 7,
across three survey waves, and what changes when the questionnaire is written in the
language of a country (*language steering*).

This repository currently holds the code. The data the scripts read (the question texts in
17 languages, the human answer distributions, the few-shot blocks) and the result files
behind every figure and table will be added when the paper is published.

## Relation to the AAAI 2026 repository

The conference repository, <https://github.com/nlply/global-opinion-alignment>, holds the
code, data and results of the conference paper: the no-steering (`code/RQ1`) and steering
(`code/RQ2`) runners, the WVS loader and response matcher (`utils/`), the wave-7
questionnaire in nine languages and the master questionnaire of waves 5–7
(`dataset/questions/`), the few-shot blocks (`inputs/few_shot/`) and the result files of
the conference experiments (`results/`). Nothing there is duplicated here; this repository
imports it (see Setup).

What this repository adds for the journal version:

| Path | What it holds |
|---|---|
| `code/config.py`, `code/jobs.py`, `code/prompts.py`, `code/parse.py` | the shared configuration, the list of prompts of the journal experiments, prompt construction (ported from the conference `utils/common.py`) and response parsing (faithful to its `match_llm_str()`) |
| `code/run_api.py`, `code/run_llm.py`, `code/run_langonly.py` | runners for the journal prompts: the GPT models through the OpenAI Batch API, the open-weight models with vLLM, and the country-blind prompts of Section 5.2 |
| `code/run_aaai_batch.py`, `code/run_anon_batch.py`, `code/run_anon_llm.py`, `code/aaai_anon.py` | re-runs of the conference-format conditions for GPT-4.1 and GPT-5, and the two anonymous few-shot conditions (`few-shot-anon`, `language_few-shot-anon`) for all six LLMs |
| `code/score.py`, `code/speaker_rank.py` | alignment scores, the within-country difference-in-differences, and the speaker-group gains of Section 5.2 |
| `code/activations.py`, `code/merge_shards.py`, `code/l1dist.py` | the residual-stream dumps and the layer-wise L1 distances of Section 5.3 |
| `code/questionnaire.py`, `code/lang_assets.py`, `code/fewshot.py`, `code/ground_truth.py` | building the inputs: a target-language questionnaire from an official WVS country PDF (eight languages beyond the nine of the conference paper), the steering-prompt boilerplate for those languages, the few-shot blocks, and the human answer distributions per country and per interview-language subgroup |
| `code/*.json` | the language instructions, persona slots and per-language assets the prompts are built from |
| `plot/` | one script per figure or table of the paper; `paper_style.py` holds the shared loaders and styling |

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/nlply/global-opinion-alignment ../global-opinion-alignment
```

The conference repository is imported for its WVS loader, response matcher, few-shot
blocks and `dataset/questions/`. Then download the WVS data files and put them in
`../global-opinion-alignment/dataset/wvs/`:

* `WV7.csv`: [WVS wave 7 (2017-2022)](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp), cross-national data set, CSV;
* `WV5.csv`, `WV6.csv` (only for Figure 3): the wave 5 and wave 6 cross-national data sets.

WVS data are free for non-profit use after registration, require a citation in every
publication and must not be redistributed, which is why they are not in this repository
([conditions of use](https://www.worldvaluessurvey.org/AJDownloadLicense.jsp)).

Paths can be moved with environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `AAAI_REPO` | `../global-opinion-alignment` | clone of the conference repository |
| `WVS_DIR` | `$AAAI_REPO/dataset/wvs` | directory with `WV5.csv`, `WV6.csv`, `WV7.csv` |
| `AAAI_RESULTS` | `results/aaai` | conference-format results |
| `OPENAI_API_KEY` | – | for the GPT runners |
| `HF_HOME`, `HF_TOKEN_FILE` | `~/.cache/huggingface` | weights and token for the open-weight models (Llama 3 is gated) |
| `HIDDEN_DIR` | `hidden/` | where `code/activations.py` writes the residual-stream dumps (tens of GB per model) |
| `GPU_HOSTS` | `localhost` | hosts for the vLLM runs |

## Figures and tables

Every figure and table of the paper is produced by one script from result files; none of
them needs a model to be run again. The result files will be added to `results/` on
publication, together with `data/`, `questions/` and `few_shot/`.

| In the paper | Script | Reads |
|---|---|---|
| Figure 1 | `python plot/demo.py` | `results/gpt-5/parsed.csv`, `data/ground_truth.json` |
| Figure 2 (a–c), Figure 6 | `python plot/default_alignment.py` | `results/aaai/RQ1`, `data/ground_truth.json` |
| Figure 3 | `python plot/temporal.py` | `results/aaai/RQ1`, `WV5/6/7.csv`, `WV567.jsonl` |
| Table 1 | `python plot/steering_table.py` | `results/aaai/RQ1`, `results/aaai/RQ2`, the steering tables of the conference paper, `WV7.csv` |
| Figure 4, Table 2 | `python plot/speaker_gain.py`, `python plot/speaker.py` | `results/speaker_rank.csv`, `WV7.csv` |
| Figure 5 | `python plot/l1dist.py` | `results/l1dist.csv` |
| Table 4 | `python plot/sensitivity.py` | `results/aaai/RQ1`, `results/aaai/Discussion/sensitivity` |

Table 3 lists the WVS questions used for the internal-consistency check; the rates in the
text are read off `results/aaai/RQ1`.

The scores are rebuilt from the parsed model answers by

```bash
python code/score.py --model gpt-5        # -> results/gpt-5/{scores,switch,fewshot,foreign_scores,lean,did}.csv
python code/speaker_rank.py               # -> results/speaker_rank.csv
```

and `results/l1dist.csv` by `code/activations.py` → `code/merge_shards.py` → `code/l1dist.py`,
which needs the open-weight models and about 70 GB of disk per model for the dumps.

## Running the experiments

| Script | What it runs |
|---|---|
| `code/run_api.py submit\|status\|collect --model gpt-4.1` | the journal prompts (`code/jobs.py`) for a GPT model through the OpenAI Batch API → `results/<model>/raw.csv` |
| `code/run_llm.py --model qwen2.5-72b --tp 2` | the same prompts for an open-weight model with vLLM |
| `code/run_langonly.py` | the country-blind prompts in each language for Section 5.2 |
| `code/run_aaai_batch.py`, `code/run_anon_batch.py`, `code/run_anon_llm.py` | the conference-format conditions of Table 1 and Table 4 (`results/aaai/`), including the two anonymous few-shot conditions |
| `code/activations.py --model qwen2.5-72b` | the last-token residual stream of every prompt condition, for Section 5.3 |

`code/fewshot.py`, `code/ground_truth.py`, `code/lang_assets.py` and `code/questionnaire.py`
build the inputs (`few_shot/`, `data/`, the steering-prompt boilerplate and the language
versions in `questions/`) and are documented in their module docstrings, as is every other
script.

## Citation

```bibtex
@inproceedings{liu2026alignment,
  title     = {On the Alignment of Large Language Models with Global Human Opinion},
  author    = {Liu, Yang and Kaneko, Masahiro and Chu, Chenhui},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {44},
  pages     = {37673--37681},
  year      = {2026}
}
```

The journal version will be added here once it is published. If you use the human
answer distributions or the questions, please also cite the World Values Survey:
Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano, J.,
Lagos, M., Norris, P., Ponarin, E. & Puranen, B. (2024). *World Values Survey Wave 7
(2017-2022) Cross-National Data-Set*, version 6.0.0. World Values Survey Association.
doi:10.14281/18241.24.

## License

The code is released under the Apache License 2.0 (see `LICENSE`).
