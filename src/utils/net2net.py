"""Utilities for Net2Net transformations."""

from collections import Counter
from typing import Optional
import random as _random

import torch


def widen_linear(
    layer: torch.nn.Linear,
    next_layer: torch.nn.Linear,
    add_units: int,
    rng: Optional[_random.Random] = None,
) -> None:
    """
    Function-preserving *widening* for two adjacent Linear layers.

    Math: duplicate neurons and split their outgoing weights.
    If neuron j has activation h_j and outgoing column v_j, replicating it r times
    and setting each outgoing column to v_j/(r+1) keeps
        sum_{copies} v_j/(r+1) * h_j  = v_j * h_j,
    so the network function is unchanged at init.
    """
    if add_units <= 0:
        return  # caller guarantees no shrink; equal width needs no action

    rng = _random if rng is None else rng
    with torch.no_grad():
        W_in = layer.weight        # shape: (h_new, in_features)
        b_in = layer.bias          # shape: (h_new,)
        W_out = next_layer.weight   # shape: (out_features, h_new)

        h_new = W_in.shape[0]
        h_old = h_new - add_units

        # choose existing neurons to replicate
        src_idx = [rng.randrange(h_old) for _ in range(add_units)]

        # split outgoing columns once per source
        replication_count = Counter(src_idx)
        for j, c in replication_count.items():
            W_out[:, j].div_(c + 1)

        # write replicas into rows [h_old, ..., h_new-1]
        write_row = h_old
        for j in src_idx:
            W_in[write_row].copy_(W_in[j])
            b_in[write_row].copy_(b_in[j])
            W_out[:, write_row].copy_(W_out[:, j])
            write_row += 1
