# Double Descent under Structured Label Noise
*A theoretical & empirical study in two-layer neural networks (ReLU) with class-conditional / structured label noise.*

**Author:** Visho Malla Oli  
**Advisor:** Dr. Hailin Sang

This repo accompanies the paper *“Double Descent under Structured Label Noise: A Theoretical & Empirical Study in Two-Layer Neural Networks.”*  
We simulate a teacher–student setup and show how structured label noise amplifies the interpolation peak and lifts the second-descent floor. We also show that modest $L_2$ regularization mitigates the spike.

## What’s inside
- `paper/` – LaTeX source for the paper (place the `main.tex` I provided here). Figures are saved into `paper/figs/`.
- `scripts/` – Minimal Python to reproduce the two plots + CSVs:
  - **Figure 1**: test error vs. model size across noise levels (`pi = 0, 0.1, 0.3`)
  - **Figure 2**: effect of $L_2$ regularization at fixed noise (`pi = 0.3`)
- `results/` – CSV outputs to support PGFPlots/tables.

## Quick start
```bash
# clone and enter
git clone https://github.com/<your-username>/double-descent-structured-noise.git
cd double-descent-structured-noise

# (optional) create a venv
python3 -m venv .venv && source .venv/bin/activate

# install deps
pip install -r requirements.txt

# generate figures + CSVs
python scripts/generate_figures.py


@misc{mallaoli2025ddnoise,
  author = {Visho Malla Oli},
  title  = {Double Descent under Structured Label Noise: A Theoretical \& Empirical Study in Two–Layer Neural Networks},
  year   = {2025},
  note   = {Preprint}
}
