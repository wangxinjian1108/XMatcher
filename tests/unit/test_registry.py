import pytest
from xmatcher.core.registry import register, get_matcher_cls, list_methods, _REGISTRY


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Registry is global; snapshot/restore around each test."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def test_register_decorator_assigns_method_name():
    @register("foo")
    class Foo:
        pass
    assert Foo.method_name == "foo"
    assert get_matcher_cls("foo") is Foo


def test_register_duplicate_raises():
    @register("bar")
    class Bar:
        pass
    with pytest.raises(KeyError, match="already registered"):
        @register("bar")
        class Bar2:
            pass


def test_get_unknown_raises_with_available_list():
    @register("baz")
    class Baz:
        pass
    with pytest.raises(KeyError, match=r"Unknown matcher 'qux'.*Available.*baz"):
        get_matcher_cls("qux")


def test_list_methods_returns_sorted_names():
    @register("zeta")
    class Z: pass
    @register("alpha")
    class A: pass
    assert list_methods() == ["alpha", "zeta"]


def test_lightglue_registers_on_methods_import():
    """Importing xmatcher.methods triggers @register('lightglue')."""
    from xmatcher.core.registry import _REGISTRY
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    try:
        # Force reimport of the methods package.
        import importlib, sys
        if "xmatcher.methods" in sys.modules:
            del sys.modules["xmatcher.methods"]
        if "xmatcher.methods.lightglue" in sys.modules:
            del sys.modules["xmatcher.methods.lightglue"]
        importlib.import_module("xmatcher.methods")
        assert "lightglue" in _REGISTRY
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)
