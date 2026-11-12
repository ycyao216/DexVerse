"""Name-only compatibility for recorded data; these are never Gym aliases.

Keep actions, states, asset bindings and benchmark revisions untouched. Renaming
a task does not make data from a different physics revision compatible.
"""

FOLDABLE_PHONE_TASK = "Dexverse-OpenFoldablePhone-v0"

# Historical recording IDs, including the pre-DexVerse collection spelling.
RETIRED_RECORDING_TASKS = {
    "Dexverse-OpenHuaweiPhone-v0": FOLDABLE_PHONE_TASK,
    "Dexverse-FixateThenManipulate-OpenHuaweiPhone-v0": FOLDABLE_PHONE_TASK,
    "Dexbench-OpenHuaweiPhone-v0": FOLDABLE_PHONE_TASK,
    "Dexbench-FixateThenManipulate-OpenHuaweiPhone-v0": FOLDABLE_PHONE_TASK,
}


def recorded_task_name(name: str | None) -> str | None:
    """Translate a known recorded name, leaving all other identities alone."""
    if name is None:
        return None
    bare = name.split(":")[-1]
    if bare == FOLDABLE_PHONE_TASK:
        return bare
    return RETIRED_RECORDING_TASKS.get(bare, name)


def normalize_recording_identity(payload: dict) -> dict:
    """Return a shallow copy only when a phone recording needs a name update.

    The original labels are retained as provenance. No episode data is copied
    or edited, and inconsistent task/env_name metadata is never repaired by
    guessing which task was intended.
    """
    original = {key: payload[key] for key in ("task", "env_name") if payload.get(key) is not None}
    normalized = {key: recorded_task_name(value) for key, value in original.items()}
    if FOLDABLE_PHONE_TASK in normalized.values() and len(set(normalized.values())) != 1:
        raise ValueError(f"Conflicting recording task identities: {original!r}")
    if normalized == original:
        return payload
    result = dict(payload)
    result.update(normalized)
    result.setdefault("recorded_task_identity", original)
    return result


def recording_output_label(label: str) -> str:
    """Use the new name in export labels, retaining input paths as provenance."""
    for old, new in RETIRED_RECORDING_TASKS.items():
        label = label.replace(old, new)
    return label
