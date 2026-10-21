"""One checkpoint, loaded once, whatever the mode.

A 32B model in 4-bit fills most of a 40 GB card. Two modes that each build
their own runtime load it twice and the run dies after the first mode has
already generated --- which is exactly what happened, twice, and the second
time only because an assignment later in the constructor quietly reset the
shared runtime to ``None``. A test is cheaper than another forty minutes of
download.
"""

from __future__ import annotations

from spatial_reasoning.functional_parser import FunctionalQwenParser
from spatial_reasoning.llm_parser import QwenQuestionParser
from spatial_reasoning.qwen_runtime import QwenRuntime, RuntimeConfig
from spatial_reasoning.regions import RegionConfig


class _Sentinel(QwenRuntime):
    """A runtime that refuses to load, so a second load is loud."""

    def ensure_loaded(self) -> None:          # pragma: no cover - never called
        raise AssertionError("the shared runtime was asked to load the model")

    @property
    def device(self) -> str:
        return "cpu"


def _runtime() -> _Sentinel:
    return _Sentinel(RuntimeConfig(model_name="stub", device="cpu"))


def test_the_flat_parser_keeps_the_runtime_it_was_given():
    shared = _runtime()
    parser = QwenQuestionParser(model_name="stub", runtime=shared)
    assert parser._runtime is shared


def test_the_functional_parser_keeps_the_runtime_it_was_given():
    shared = _runtime()
    parser = FunctionalQwenParser(model_name="stub", runtime=shared,
                                  region_config=RegionConfig(layout="nuscenes_qa"))
    assert parser.runtime is shared


def test_each_parser_still_builds_its_own_when_none_is_shared():
    """The single-mode entry points must keep working unchanged."""
    flat = QwenQuestionParser(model_name="stub")
    assert flat._runtime is None          # built lazily, on first generation
    typed = FunctionalQwenParser(model_name="stub")
    assert typed.runtime is not None and typed.runtime.config.model_name == "stub"
