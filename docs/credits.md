# Attribution

The experiment code implements the architecture-matched baseline objectives
described in the SPRL paper. In particular, Balanced is a reward-prediction
network with a representation-distribution MMD penalty, and LDM is the
SPRL reward/latent-transition backbone with zero propensity penalty. These
are the study's specified implementations, not claims of exact reproduction
of every variant in the associated representation-learning literature.

The IHDP covariate source is the
[CEVAE authors' repository](https://github.com/AMLab-Amsterdam/CEVAE), associated
with *Causal Effect Inference with Deep Latent-Variable Models* (Louizos et al.,
NeurIPS 2017). The HalfCheetah source is maintained through
[Farama Minari](https://minari.farama.org/). Follow the data sources' citation
and usage terms when using them.

Release organization was informed by
[NeuralCSA](https://github.com/DennisFrauen/NeuralCSA),
[SharpCausalSensitivity](https://github.com/DennisFrauen/SharpCausalSensitivity),
and the [research-code release checklist](https://github.com/paperswithcode/releasing-research-code).
Their implementations were not copied into this repository.
