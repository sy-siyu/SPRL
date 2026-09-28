# SPRL — Sensitivity-Preserving Representation Learning

Code for **When Do Learned State Representations Break Sensitivity Analysis?**
(accepted at NeurIPS 2026).

**Authors:** Siyu Wang, Xiaocong Chen, Quan Z. Sheng, and Lina Yao.

SPRL learns state representations with reward/latent-transition prediction and
a kernel-weighted propensity-homogeneity penalty. This release contains the
camera-ready experiment configuration: a frozen neural propensity teacher,
the original SPRL objective, and `lambda_prop=1` for the main comparisons.

## Quick start

Use Python 3.13 and run commands from the repository root. The release was
checked on CPU; package versions are pinned below.

```bash
git clone https://github.com/sy-siyu/SPRL.git
cd SPRL
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Reconstruct the published main tables from all saved seed-level results.
python scripts/render_results.py --check --output outputs/tables

# Known-answer, implementation, and saved-result checks; no external data.
python -m unittest discover -s tests -v

# Short end-to-end training checks, deliberately smaller than the paper runs.
python -m experiments.setting_a --smoke --output outputs/smoke_a
python -m experiments.setting_b --smoke --output outputs/smoke_b
```

## Repository contents

| Directory | Contents |
|---|---|
| `representations/` | SPRL, frozen propensity teacher, AE, VAE, contrastive, and Balanced baselines |
| `dgp/` | Synthetic generators, benchmark loading, finite constructions |
| `estimators/` | Samplewise interval evaluator and separately labelled finite scalar diagnostic |
| `evaluation/` | Fitted action-sensitivity readouts |
| `experiments/` | Training/evaluation entry points, controlled ablations, exact examples |
| `configs/` | Main experiment parameters and seed ranges |
| `results/` | Unrounded, seed-level numerical results; no observations or checkpoints |
| `scripts/` | Saved-result table reconstruction |
| `tests/` | Fast mathematical and implementation checks |
| `docs/` | Data preparation and appendix reproduction commands |

The baselines are the architecture-matched implementations specified in the
paper. LDM uses the same predictor architecture as SPRL with zero propensity
penalty. The unit-sphere ablation is kept separate from the main method.

## Train and evaluate the main experiments

```bash
# Table 1 / Setting A: seeds 1–100, three confounding strengths.
python -m experiments.setting_a --first-seed 1 --last-seed 100 --output outputs/setting_a

# Table 2 and the lambda sweep / Setting B: seeds 1–100.
python -m experiments.setting_b --first-seed 1 --last-seed 100 --output outputs/setting_b

# Table 3 and full semi-synthetic results: obtain data as described below.
python -m experiments.benchmarks --benchmark ihdp --first-seed 101 --last-seed 200 \
  --ihdp-data data/ihdp_npci_1.csv --output outputs/ihdp
python -m experiments.benchmarks --benchmark halfcheetah --first-seed 101 --last-seed 200 \
  --output outputs/halfcheetah
```

Main protocol settings are recorded in `configs/`; smoke runs use a separately recorded,
reduced configuration and are not paper results. Full runs are CPU-intensive:
each includes hundreds of representation fits. Start with a smoke run or a
single seed to estimate runtime on your machine. Training-generated outputs
and checkpoints belong under the ignored `outputs/` directory.
The standalone main commands use 10 PyTorch compute threads and 14 interop
threads; synthetic smoke checks use one compute thread.

For exact constructions, controlled objective/dimension ablations, teacher
perturbations, and representation diagnostics, see
[Appendix experiments](docs/appendix_experiments.md).

## Data

Synthetic experiments generate their data locally. IHDP and HalfCheetah source
datasets are obtained separately; no dataset download occurs during tests.
See [Data preparation](docs/data.md) for sources, required versions, and splits.
Install `requirements-benchmarks.txt` only when running HalfCheetah.

## Reading the results

Setting A and IHDP evaluate empirical inclusion of a known target using
sign-aware samplewise HT-IPW intervals. These are plug-in sensitivity
intervals, not finite-sample confidence intervals. The returned interval is an
outer envelope rather than the sharp normalized MSM solution. The finite
one-step example separately implements the normalized reward-distribution LP.

Setting B and HalfCheetah report fitted median and p95 action-sensitivity
ratios. They depend on the specified readout and clipping protocol and are
not population worst-case sensitivity estimates. Latent-transition MSE uses
each encoder's own coordinate system.

The saved results preserve the paper's tradeoffs: in Setting A, SPRL includes
the target in all 100 seeds at each strength, with mean width ratios to Raw
of 1.001, 1.016, and 1.053. In Setting B, its mean fitted p95 ratios are
2.66, 2.32, and 2.48 versus LDM's 4.92, 4.76, and 5.01. See
[Results](results/README.md) for the full seed-level data and aggregation rules.

Saved-result reconstruction does not rerun training. Numerical training
reproduction can depend on library versions, floating-point backends, and
thread settings. The supplied checks test execution and stated numerical
identities; the full training suite is run by the commands above.

## Citation and license

Please cite the accepted paper:

```bibtex
@inproceedings{wang2026sprl,
  title     = {When Do Learned State Representations Break Sensitivity Analysis?},
  author    = {Wang, Siyu and Chen, Xiaocong and Sheng, Quan Z. and Yao, Lina},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Accepted; proceedings details forthcoming}
}
```

The official proceedings link and remaining bibliographic details will be
added when available.

The code is released under the [MIT License](LICENSE). External datasets
retain their own terms; this repository does not redistribute them.
