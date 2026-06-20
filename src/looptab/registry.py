"""Registry: resolve string names from configs to objects."""

from .models.controls import FFMatched, UntiedStack, UntiedStackMatched
from .models.trm import TRM

MODEL_REGISTRY = {
    "trm": TRM,
    "ff_matched": FFMatched,
    "untied_stack": UntiedStack,
    "untied_matched": UntiedStackMatched,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
