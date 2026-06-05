from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise KeyError(f"Matcher '{name}' already registered")
        cls.method_name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_matcher_cls(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown matcher '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_methods() -> list[str]:
    return sorted(_REGISTRY)
