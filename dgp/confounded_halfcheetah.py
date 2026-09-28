"""Load Minari HalfCheetah episodes for the semi-synthetic study.

The dataset must already be installed; this function never downloads data.
Synthetic binary assignment is defined separately in the benchmark runner.
Original rewards/transitions come from the continuous-control source data.
"""

import numpy as np


def load_halfcheetah_episodes(dataset_name="mujoco/halfcheetah/medium-v0",
                            max_episodes=None):
    """Load observations, next observations, continuous actions, and rewards."""
    try:
        import minari
    except ImportError as exc:
        raise ImportError("Install requirements-benchmarks.txt for HalfCheetah") from exc
    dataset = minari.load_dataset(dataset_name)
    episodes = []
    for i, ep in enumerate(dataset.iterate_episodes()):
        if max_episodes and i >= max_episodes:
            break
        episodes.append({
            "s": ep.observations[:-1].astype(np.float32),
            "s_next": ep.observations[1:].astype(np.float32),
            "a_cont": ep.actions.astype(np.float32),
            "r": ep.rewards.astype(np.float32),
        })
    return episodes
