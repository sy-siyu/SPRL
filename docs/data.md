# Data preparation

## Synthetic Settings A and B

No download is needed. `dgp/unified_cmdp.py` generates states, binary actions,
rewards, and transitions from the experiment seeds. Setting A uses the
reward-aligned policy weights specified in `configs/setting_a.json`; Setting B
uses the overlapping five-step design in `configs/setting_b.json`.

## IHDP-derived semi-synthetic bandit

Obtain `ihdp_npci_1.csv` from the
[CEVAE author's IHDP source](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP/csv),
observing the source dataset's terms, and place it at `data/ihdp_npci_1.csv`.
The loader expects 747 rows and 30 columns. Only columns **5:30** (zero-based)
are covariates; the original treatment and four outcome fields are excluded.

All 747 covariate rows are standardized to define the finite source pool.
The study draws 5,000 training and 10,000 fresh evaluation observations with
replacement from that same pool. They are independent draws, not a held-out
subject study. The binary hidden variable, actions, and rewards are generated
locally. The raw propensity uses the exact conditional hidden-variable
distribution on the shifted empirical support.

## HalfCheetah-derived semi-synthetic study

Install the optional dependencies:

```bash
pip install -r requirements-benchmarks.txt
```

Obtain the Minari dataset **`mujoco/halfcheetah/medium-v0`** using the
[Minari documentation](https://minari.farama.org/). For an explicit download:

```bash
python -c "import minari; minari.download_dataset('mujoco/halfcheetah/medium-v0')"
```

The runner loads the first 1,000 episodes of length 1,000. This is the Minari
HalfCheetah-v5 dataset, not the older D4RL HDF5 file. It uses the default
17-coordinate observation convention documented by
[Gymnasium](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).

For each seed, 50 complete episodes train the representation, 10 fit the
readout, and 10 score it. These sets are disjoint. A state standardizer is fitted
on training episodes only. The injected hidden variable is constant within
each episode; synthetic action probabilities use the numeric coordinates and
coefficients in the released configuration. The injection shifts coordinates
0, 1, 8, 11, and 12 in current and next states without changing source rewards.

The binary actions are synthetic labels over logged continuous-control data;
they did not generate the source physics transitions. This study evaluates
fitted assignment sensitivity and prediction, not binary-policy value.

## Distribution boundary

No IHDP source rows, Minari episodes, learned embeddings, identifiers, or
trained checkpoints are bundled. `results/` contains numerical experiment
summaries for every reported seed. Data and generated models are ignored by
Git. Keep any separately obtained source data under `data/` or in Minari's
local dataset storage, subject to the corresponding data terms.
