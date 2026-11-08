"""Load persisted model bundles; fall back to deterministic in-memory training if absent."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import joblib

from finguard.models.bundle import ModelBundle
from finguard.models.train import save_bundles, train_all

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_cache: dict[str, dict[str, ModelBundle]] = {}


def get_bundles(model_dir: Path, seed: int = 42, autotrain: bool = True) -> dict[str, ModelBundle]:
    """Return ``{model_name: bundle}`` for ``model_dir``.

    Missing artefacts fail safe: either retrain deterministically (``autotrain``) or raise.
    """
    key = str(model_dir.resolve())
    with _lock:
        if key in _cache:
            return _cache[key]
        files = sorted(model_dir.glob("*.joblib")) if model_dir.exists() else []
        if files:
            bundles = {b.name: b for b in (joblib.load(f) for f in files)}
        elif autotrain:
            logger.warning("no model artefacts in %s; training deterministic baselines", model_dir)
            bundles = train_all(seed)
            try:
                save_bundles(bundles, model_dir)
            except OSError:
                logger.exception("could not persist models; continuing in memory")
        else:
            raise FileNotFoundError(f"No model artefacts in {model_dir}. Run `make train`.")
        _cache[key] = bundles
        return bundles


def clear_cache() -> None:
    """Drop cached bundles (used by tests)."""
    with _lock:
        _cache.clear()
