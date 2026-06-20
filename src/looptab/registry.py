"""Registry: resolve string names from configs to objects."""

from .models.trm import TRM
from .models.controls import FFMatched

MODEL_REGISTRY = {
    "trm": TRM,
    "ff_matched": FFMatched,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
