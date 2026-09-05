# PRAHARI-P2-NegativeResult

**What Provenance Buys: A Negative Result on Label-Free Accuracy Estimation**

Author: Amit Dua (Yushu Excellence Technologies Pvt. Ltd.)  
Tier: C · Venue aim: Workshop (not ICCV)

## Claim

Average Confidence wins on batch accuracy; provenance stratification only buys per-path estimates.

Parent platform: https://github.com/amitduabits/PRAHARI

## Layout

```
paper/main.tex          draft
paper/references.bib    verified BibTeX
code/                   experiment harness + prresearch package
data/                   fixtures and how to get live data
results/                generated locally (gitignored JSON/JSONL)
figs/                   figures
```

## Reproduce (synthetic, <5 min)

```bash
cd code
pip install -r requirements.txt
python -m pytest -q
python -c "from prresearch.p2_fallback.experiment import main; main()"
```

## Reproduce (MEASURED P1/P4)

From the parent PRAHARI tree, not this repo:

```bash
cd 02_Code/prahari
python scripts/instrument.py all --seconds 8 --frames 6 --k-frames 6 --seed-n 24 --k 1 2 4
```

Live 24 h RTSP is `09_Research/scripts/capture_live.py` in the parent tree.
Do not commit `.env`, raw video, or `*.jsonl`.

## Licence

Paper: CC BY 4.0. Code: CC BY-SA 4.0.
