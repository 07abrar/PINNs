"""Utilities for Net2Net transformations."""

from __future__ import annotations

from collections import Counter
from typing import Optional
import random as _random

import torch


def widen_linear(
    L: torch.nn.Linear,
    Lnext: torch.nn.Linear,
    add_k: int,
    rng: Optional[_random.Random] = None,
) -> None:
    """Function-preserving widening of two adjacent ``nn.Linear`` layers.

    This operates on layers that already have the *target* width ``h_old + add_k``.
    The first ``h_old`` neurons are assumed to contain weights copied from a
    smaller pretrained model. The newly added ``add_k`` neurons are populated by
    replicating existing neurons and splitting the outgoing weights of ``Lnext``.
    """

    rng = _random if rng is None else rng

    W, b = L.weight.data, L.bias.data
    Wn = Lnext.weight.data

    h_new, in_dim = W.shape
    h_old = h_new - add_k
    out_dim, _ = Wn.shape

    # choose source indices to replicate
    src_idx = [rng.randrange(h_old) for _ in range(add_k)]

    # counts per source for splitting outgoing weights
    cnt = Counter(src_idx)
    for j, c in cnt.items():
        Wn[:, j] = Wn[:, j] / (c + 1)

    # add replicas in L and split columns in Lnext
    col_ptr = h_old
    for j in src_idx:
        W[col_ptr] = W[j]
        b[col_ptr] = b[j]
        Wn[:, col_ptr] = Wn[:, j]
        col_ptr += 1
