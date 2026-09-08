"""Load only the MACE Body Adapter tensors required by inference."""

from collections import Counter
from collections.abc import Mapping


BODY_ADAPTER_PREFIX = "pose_patch_embedding."
CHECKPOINT_PREFIXES = (
    "module.pipe.animate_adapter.",
    "pipe.animate_adapter.",
    "module.animate_adapter.",
    "animate_adapter.",
)


def _normalise_key(name: str) -> str:
    for prefix in CHECKPOINT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def select_body_adapter_state_dict(state_dict: Mapping):
    """Return the pose projection and retain Wan's original face modules."""
    selected = {
        name: value
        for original_name, value in state_dict.items()
        if (name := _normalise_key(original_name)).startswith(BODY_ADAPTER_PREFIX)
    }
    if not selected:
        raise ValueError("Checkpoint contains no Body Adapter tensors.")
    return selected


def animate_adapter_component_counts(state_dict: Mapping) -> dict[str, int]:
    """Count selected tensors by their top-level adapter component."""
    counts = Counter(_normalise_key(name).split(".", 1)[0] for name in state_dict)
    return dict(sorted(counts.items()))
