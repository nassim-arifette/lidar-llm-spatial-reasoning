"""Answering a spatial question versus computing its answer.

Perception is fixed throughout: object lists are consumed, never computed, so
every comparison is about reasoning rather than detection.

The central experiment runs ONE frozen language model two ways on the same
questions and the same objects --- it answers directly (``direct_qa``), or it
writes a program (``llm_parser``) that a deterministic executor runs
(``dsl``). ``compare_llm_modes`` pairs them question by question on the frozen
example set in ``manifest``.

Supporting pieces: ``question_parser`` is the deterministic rule grammar used
as a reference and a baseline, ``verify_executor`` checks the executor with
reference programs and no model involved, ``analyse_failures`` triages what is
left, and ``baselines``/``run_experiments`` hold the earlier trained-baseline
study. ``functional`` is a CLEVR-style typed catalogue, implemented and
deliberately not wired into any experiment.

Entry point:
``python -m spatial_reasoning.compare_llm_modes --manifest ... --split eval``
"""

from .objects import (  # noqa: F401
    CANONICAL_CATEGORIES,
    Object3D,
    Scene,
    load_nuscenes_detections,
    load_simple_scenes,
    save_simple_scenes,
)
from .regions import (  # noqa: F401
    DEFAULT_REGION_CONFIG,
    DIRECTIONS,
    Region,
    RegionConfig,
)
from .question_parser import (  # noqa: F401
    Operation,
    ParseStatus,
    Query,
    Grammar,
    RuleBasedQuestionParser,
    parse_question,
)
from .programs import ExecutionResult, SymbolicSpatialQA, execute  # noqa: F401
from .datasets import (  # noqa: F401
    AnswerVocabulary,
    HashingQuestionEncoder,
    QAItem,
    split_by_scene,
)

__all__ = [
    "CANONICAL_CATEGORIES", "Object3D", "Scene",
    "load_nuscenes_detections", "load_simple_scenes", "save_simple_scenes",
    "DEFAULT_REGION_CONFIG", "DIRECTIONS", "Region", "RegionConfig",
    "Grammar", "Operation", "ParseStatus", "Query", "RuleBasedQuestionParser",
    "parse_question",
    "ExecutionResult", "SymbolicSpatialQA", "execute",
    "AnswerVocabulary", "HashingQuestionEncoder", "QAItem", "split_by_scene",
]

__version__ = "0.1.0"
