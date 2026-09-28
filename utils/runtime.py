"""Bounded CPU parallelism for standalone experiment entry points."""

import torch


def configure_cpu(smoke=False):
    # 10/14 matches the recorded corrected Setting A/benchmark execution.
    # Use one compute thread for small installation checks.
    torch.set_num_threads(1 if smoke else 10)
    if torch.get_num_interop_threads() != 14:
        torch.set_num_interop_threads(14)
