import pytest
from pydantic import BaseModel, ValidationError
from xmatcher.core.config import RunConfig, build_matcher
from xmatcher.core.base import BaseMatcher
from xmatcher.core.registry import register, _REGISTRY
from xmatcher.core.types import _RawOutput
import torch


@pytest.fixture(autouse=True)
def _isolated_registry():
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


class _ToyParams(BaseModel):
    threshold: float = 0.1
    label: str


class _ToyMatcher(BaseMatcher):
    Params = _ToyParams
    def _setup(self): pass
    def _forward(self, pair):
        return _RawOutput(
            mkpts0=torch.zeros(0, 2), mkpts1=torch.zeros(0, 2),
            mconf=torch.zeros(0), dense=None,
        )


def _register_toy():
    """Register the toy matcher. Call from inside tests that need it
    (fixture clears registry between tests)."""
    if "toy" not in _REGISTRY:
        register("toy")(_ToyMatcher)


def test_runconfig_accepts_minimum_fields():
    _register_toy()
    cfg = RunConfig.model_validate({
        "method": "toy",
        "params": {"label": "x"},
    })
    assert cfg.method == "toy"
    assert cfg.device == "cuda"
    assert cfg.seed == 0


def test_build_matcher_routes_params_to_typed_model():
    _register_toy()
    cfg = RunConfig.model_validate({
        "method": "toy",
        "device": "cpu",
        "params": {"threshold": 0.5, "label": "x"},
    })
    m = build_matcher(cfg)
    assert isinstance(m, _ToyMatcher)
    assert m.params.threshold == 0.5
    assert m.params.label == "x"
    assert m.device == "cpu"


def test_build_matcher_rejects_unknown_method():
    cfg = RunConfig.model_validate({"method": "nope", "params": {}})
    with pytest.raises(KeyError, match="Unknown matcher"):
        build_matcher(cfg)


def test_build_matcher_rejects_invalid_params():
    _register_toy()
    cfg = RunConfig.model_validate({
        "method": "toy",
        "params": {"threshold": 0.5},   # missing required `label`
    })
    with pytest.raises(ValidationError):
        build_matcher(cfg)


def test_runconfig_rejects_unknown_device():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"method": "toy", "device": "tpu", "params": {}})
