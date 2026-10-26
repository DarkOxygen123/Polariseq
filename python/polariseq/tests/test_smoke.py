"""Phase 0 round-trip tests.

These prove the PyO3 / numpy / maturin / Python chain is end to end green on
this interpreter. They are not real compute tests — real pipelines (PCA,
neighbors, Leiden) arrive in later phases.
"""

from __future__ import annotations

import numpy as np
import polariseq
import pytest


def test_import_succeeds() -> None:
    """The package imports and exposes the canary function."""
    assert hasattr(polariseq, "triple")
    assert callable(polariseq.triple)


def test_version_is_exposed() -> None:
    """Both Python and Rust versions are present."""
    assert polariseq.__version__
    assert polariseq.core_version()


def test_versions_match() -> None:
    """Python and Rust core versions are in sync (driven from one workspace)."""
    info = polariseq.get_build_info()
    assert info["matched"], (
        f"Python version {info['python_version']!r} != "
        f"Rust core version {info['rust_core_version']!r}"
    )


def test_triple_doubles_correctly() -> None:
    """The canary: triple([1, 2, 3]) == [3, 6, 9] (round-trips through Rust)."""
    inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    out = polariseq.triple(inp)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [3.0, 6.0, 9.0, 12.0])


def test_triple_does_not_mutate_input() -> None:
    """The Rust side borrows read-only; the caller's array must be unchanged."""
    inp = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    polariseq.triple(inp)
    np.testing.assert_allclose(inp, [1.0, 2.0, 3.0])


def test_triple_handles_empty() -> None:
    """Empty input round-trips cleanly."""
    inp = np.array([], dtype=np.float64)
    out = polariseq.triple(inp)
    assert out.shape == (0,)


def test_triple_rejects_wrong_dtype() -> None:
    """float32 input must be rejected — the canary pins float64 for now."""
    inp = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises((TypeError, ValueError)):
        polariseq.triple(inp)
