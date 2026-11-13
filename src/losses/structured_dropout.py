"""
Structured sensor dropout: randomly removes whole modalities, whole body
locations, or individual channels — not just random point-masking. This
teaches the model that a person's underlying state shouldn't change just
because the sensor configuration did (real deployments lose whole sensors,
not random timesteps).
"""
from __future__ import annotations
import random


def structured_dropout(channels: list, p_modality: float = 0.25, p_location: float = 0.25,
                        p_channel: float = 0.2, rng: random.Random = None) -> list:
    rng = rng or random
    if not channels:
        return channels
    kept = list(channels)

    for m in list({c.modality for c in kept}):
        if rng.random() < p_modality:
            kept = [c for c in kept if c.modality != m]
    for l in list({c.location for c in kept}):
        if rng.random() < p_location:
            kept = [c for c in kept if c.location != l]
    kept = [c for c in kept if rng.random() >= p_channel]

    if not kept:
        kept = [rng.choice(channels)]
    return kept
