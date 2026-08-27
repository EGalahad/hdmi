import pytest

from hdmi.tasks.command import _resolve_unique


def test_resolve_unique_preserves_available_order():
    ids, names = _resolve_unique(["object_b", "object_a"], ["object_a", "object_b"], label="body")
    assert ids == [0, 1]
    assert names == ["object_a", "object_b"]


def test_resolve_unique_rejects_missing_names():
    with pytest.raises(ValueError):
        _resolve_unique("missing", ["object"], label="body")
