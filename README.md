# Opinion Alignment and Language Steering in Large Language Models

Code for the paper

> Yang Liu, Masahiro Kaneko, and Chenhui Chu.
> **Opinion Alignment and Language Steering in Large Language Models.**
> *Journal of Natural Language Processing* (under review).

This is the journal extension of our AAAI 2026 paper
*On the Alignment of Large Language Models with Global Human Opinion*
([paper](https://arxiv.org/abs/2509.01418), [code](https://github.com/nlply/global-opinion-alignment)).
The conference code, data and results stay in that repository; this one holds the code
added for the journal version. The data and result files will be added when the paper is
published.

## Layout

- `code/`: the experiment pipeline (prompt construction, model runners, scoring); every script is documented in its module docstring
- `plot/`: one script per figure or table of the paper

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/nlply/global-opinion-alignment ../global-opinion-alignment
```

Download the WVS wave 7 data file `WV7.csv` (and `WV5.csv`, `WV6.csv` for Figure 3) from
[worldvaluessurvey.org](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp) into
`../global-opinion-alignment/dataset/wvs/`. The WVS data may not be redistributed, so they
are not included. Paths can be changed with the environment variables `AAAI_REPO` and
`WVS_DIR`; the GPT runners read `OPENAI_API_KEY`.

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

## License

Apache License 2.0 (see `LICENSE`).
