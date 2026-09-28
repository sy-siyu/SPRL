# Appendix experiments

These entry points retain the paper's supplementary constructions and
ablations separately from the main SPRL configuration. Run from the repository
root. Generated outputs are written under `outputs/`.

## Finite constructions (seconds; no model training)

```bash
python -m experiments.exact_one_step
python -m experiments.exact_policy_mixtures
python -m experiments.exact_outcome_h2 --output_dir outputs/exact_outcome_h2
python -m experiments.exact_transition_h2 --output_dir outputs/exact_transition_h2
```

The one-step example reports both the paper's statewise scalar calculation
and the normalized outcome-dependent LP. The scalar calculation is an
illustration, not a general full-MSM interval estimator; it is never used by
the learned Setting A or IHDP runners. The H=2 outcome envelope and the direct
transition-row model are distinct sensitivity constructions.

## Controlled objective ablations

```bash
python -m experiments.objective_ablation --K 10 --w_u 0.3 \
  --output_dir outputs/objective_ablation
python -m experiments.objective_unit_sphere --K 10 --w_u 0.3 \
  --source_output_dir outputs/objective_ablation \
  --output_dir outputs/objective_unit_sphere
python -m experiments.unit_sphere_evaluation \
  --input outputs/objective_unit_sphere/p2_stabilized_ablation.json \
  --output outputs/objective_unit_sphere/p2_stabilized_heldout_reward.json
```

The unit-sphere run reuses the frozen propensity teacher produced by the first
command. Its geometric normalization is an ablation, not the main method.

## Latent dimension

```bash
python -m experiments.latent_dimension --K 10 --w_u 0.3 --latent_dims 2 10 \
  --output_dir outputs/latent_dimension
```

The paper's dimension-5 rows are the matched LDM/full-SPRL pair from
`objective_ablation`; dimensions 2 and 10 use the command above. All three
dimensions retain the same nominal penalty coefficient rather than retuning it.

## Frozen-representation propensity diagnostics

```bash
python -m experiments.propensity_diagnostic --K 10 --w_u 0.3 \
  --output_dir outputs/propensity_diagnostic
```

This runner trains the stated lambda sweep, freezes each encoder, and computes
quantized and local propensity-range diagnostics. The quantized calculation
uses known propensities on a finite Monte Carlo sample; it is not an exact
population essential-supremum calculation. Oracle hidden variables are used
for evaluation, not SPRL training.

## Gaussian teacher perturbations

```bash
python -m experiments.teacher_noise --K 100 --w_u 0.5 --lambda_prop 1 \
  --noise_mode per_batch --sigmas 0.02 0.05 0.10 0.20 \
  --save outputs/teacher_noise/setting_b.json
```

The zero-noise row in the paper is the main Setting B `lambda=1` result.
The specified fresh Gaussian corruption tests additive-noise sensitivity;
the optional fixed-noise mode is not part of the reported table.

## Three-hidden-stratum example

```bash
python -m experiments.three_stratum --K 30 --output_dir outputs/three_stratum
```

This small discrete-bottleneck experiment separates exact propensity
homogeneity from pairwise sensitivity preservation with three hidden strata.
It uses the experiment-specific architecture, not a substituted main SPRL
implementation. Exact odds ratios are computed by enumerating the finite law.

## Saved appendix results

`results/appendix/` contains the unrounded synthetic seed-summary JSONs for
these ablations. Paths to private working files and checkpoint bookkeeping
were removed; numerical result fields and every completed seed are retained.
`latent_dimension_2.json` and `latent_dimension_10.json` hold the corresponding
dimension studies; dimension 5 is in `objective_ablation.json`.

The release's unit tests execute the finite constructions and short main
training paths. The full appendix training commands are supplied, but were
not rerun as part of packaging. For exact saved main-table and uncertainty
reconstruction use `scripts/render_results.py`.
