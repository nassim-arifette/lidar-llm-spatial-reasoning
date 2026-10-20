"""Low-cost spatial reasoning over pre-extracted 3D detections.

This package replaces the GPT-2 answer head of the original re-implementation
for one restricted family of questions (counting, existence, comparison) and
makes the alternative measurable:

* perception is fixed -- detections are consumed, never computed;
* the question is turned into an explicit program (``question_parser``);
* the program is executed on the object list (``programs``);
* three learned baselines predict the answer directly instead (``baselines``);
* all four are evaluated under one protocol (``evaluate``), with consistency
  and robustness probes (``consistency``).

Entry point: ``python -m spatial_reasoning.run_experiments --source synthetic``
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
    "Operation", "ParseStatus", "Query", "RuleBasedQuestionParser", "parse_question",
    "ExecutionResult", "SymbolicSpatialQA", "execute",
    "AnswerVocabulary", "HashingQuestionEncoder", "QAItem", "split_by_scene",
]

__version__ = "0.1.0"
