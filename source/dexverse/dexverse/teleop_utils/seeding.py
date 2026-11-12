"""Resolve random-seed requests before Isaac Lab initializes Replicator."""

import secrets


def resolve_seed(seed: int) -> int:
    """Keep explicit seeds and turn the -1 sentinel into a nonnegative seed.

    Some Isaac Lab versions call Replicator before resolving the sentinel;
    Replicator rejects negative seeds. Use OS randomness so separate launches
    remain independent even when Python's random generator was already seeded.
    """
    if seed == -1:
        return secrets.randbelow(2**31)
    if seed < 0:
        raise ValueError("Seed must be nonnegative, or -1 to request a random seed")
    return seed
