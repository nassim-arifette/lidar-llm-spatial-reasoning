"""Question/answer items, scene-level splits, and feature encoding.

Two choices here are deliberate and are defended in the report:

1. **Splits are by nuScenes scene, never by question.** A random split over
   questions puts different questions about the *same* LiDAR sample on both
   sides of the split. A model can then memorise "in this scene there are three
   cars" and score well without reasoning. nuScenes-mini has 10 scenes, so the
   split is coarse and the resulting variance must be reported.

2. **The learned baselines never see the point cloud.** They see the same
   object list as the symbolic system. That is the whole point of the
   comparison: perception is held fixed, so any difference comes from the
   reasoning stage.
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .objects import CANONICAL_CATEGORIES, Scene

# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #

@dataclass
class QAItem:
    """One question about one sample."""

    question: str
    answer: str
    sample_token: str
    scene_token: Optional[str] = None
    template: str = "unknown"      # question family, for per-type breakdowns
    answer_type: str = "unknown"   # "count" | "binary" | other

    def __post_init__(self) -> None:
        self.answer = str(self.answer).strip().lower()
        if self.answer_type == "unknown":
            self.answer_type = infer_answer_type(self.answer)


def infer_answer_type(answer: str) -> str:
    a = str(answer).strip().lower()
    if a in ("yes", "no"):
        return "binary"
    if re.fullmatch(r"\d+", a):
        return "count"
    return "other"


def load_qa_json(path: str, scene_lookup: Optional[Dict[str, str]] = None) -> List[QAItem]:
    """Load a QA file.

    Accepts either a bare list of records or ``{"questions": [...]}`` (the
    nuScenes-QA layout). Field names are matched leniently because the released
    files are not consistent between splits.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload.get("questions", payload) if isinstance(payload, dict) else payload

    items: List[QAItem] = []
    for r in records:
        token = r.get("sample_token") or r.get("token") or r.get("sample")
        if token is None:
            continue
        items.append(QAItem(
            question=r.get("question") or r.get("text") or "",
            answer=r.get("answer", ""),
            sample_token=token,
            scene_token=r.get("scene_token") or (scene_lookup or {}).get(token),
            template=r.get("template_type") or r.get("question_type") or "unknown",
        ))
    return items


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def split_by_scene(
    items: Sequence[QAItem],
    scenes: Dict[str, Scene],
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Dict[str, List[QAItem]]:
    """Partition items so that no nuScenes scene appears in two splits.

    Items whose scene is unknown fall back to their ``sample_token`` as a
    pseudo-scene. That is weaker than a true scene split and is flagged in the
    returned ``"warnings"`` entry so the report cannot quietly overstate the
    protocol.
    """
    rng = np.random.default_rng(seed)

    group_of: Dict[str, str] = {}
    unknown = 0
    for it in items:
        group = it.scene_token or (scenes.get(it.sample_token).scene_token
                                   if it.sample_token in scenes else None)
        if group is None:
            group = f"sample::{it.sample_token}"
            unknown += 1
        group_of[it.sample_token] = group

    groups = sorted({g for g in group_of.values()})
    rng.shuffle(groups)

    n = len(groups)
    n_train = max(1, int(round(train_fraction * n)))
    n_val = max(1, int(round(val_fraction * n))) if n - n_train > 1 else 0
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])

    out: Dict[str, List[QAItem]] = {"train": [], "val": [], "test": []}
    for it in items:
        g = group_of[it.sample_token]
        key = "train" if g in train_g else ("val" if g in val_g else "test")
        out[key].append(it)

    out["warnings"] = []  # type: ignore[assignment]
    if unknown:
        out["warnings"].append(  # type: ignore[union-attr]
            f"{unknown}/{len(items)} items had no scene id and were grouped by "
            f"sample token; the split is weaker than a true scene split"
        )
    if n < 5:
        out["warnings"].append(  # type: ignore[union-attr]
            f"only {n} scene groups available; per-split variance will be large"
        )
    return out


# --------------------------------------------------------------------------- #
# Answer vocabulary
# --------------------------------------------------------------------------- #

class AnswerVocabulary:
    """Closed answer set, built from the training split only.

    Built on train only so that a test answer the model never saw counts as an
    error rather than being quietly added to the label space.
    """

    def __init__(self, answers: Iterable[str], max_count: int = 20) -> None:
        seen: List[str] = []
        for a in answers:
            a = str(a).strip().lower()
            if a not in seen:
                seen.append(a)
        # Always provide the small integers and the two booleans, so a count
        # that is plausible but absent from train is still predictable.
        for extra in ["yes", "no"] + [str(i) for i in range(max_count + 1)]:
            if extra not in seen:
                seen.append(extra)
        self.answers: List[str] = seen
        self.index: Dict[str, int] = {a: i for i, a in enumerate(self.answers)}

    def __len__(self) -> int:
        return len(self.answers)

    def encode(self, answer: str) -> int:
        return self.index.get(str(answer).strip().lower(), -1)

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.answers):
            return self.answers[idx]
        return ""

    def majority(self, items: Sequence[QAItem]) -> str:
        counts: Dict[str, int] = {}
        for it in items:
            counts[it.answer] = counts.get(it.answer, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else "no"


# --------------------------------------------------------------------------- #
# Feature encoding
# --------------------------------------------------------------------------- #

#: Per-object feature layout, documented because the report quotes the
#: dimensionality: one-hot class (10) + x, y, z (3) + w, l, h (3)
#: + sin/cos yaw (2) + range (1) + sin/cos bearing (2) + score (1) = 22.
OBJECT_FEATURE_DIM = len(CANONICAL_CATEGORIES) + 12

_CATEGORY_INDEX = {c: i for i, c in enumerate(CANONICAL_CATEGORIES)}

#: Normalisation constants: nuScenes detection range is +/- 50 m.
_POS_SCALE = 50.0
_SIZE_SCALE = 10.0


def encode_object(obj) -> np.ndarray:
    feat = np.zeros(OBJECT_FEATURE_DIM, dtype=np.float32)
    idx = _CATEGORY_INDEX.get(obj.category)
    if idx is not None:
        feat[idx] = 1.0
    base = len(CANONICAL_CATEGORIES)
    feat[base + 0] = obj.center[0] / _POS_SCALE
    feat[base + 1] = obj.center[1] / _POS_SCALE
    feat[base + 2] = obj.center[2] / _SIZE_SCALE
    feat[base + 3] = obj.size[0] / _SIZE_SCALE
    feat[base + 4] = obj.size[1] / _SIZE_SCALE
    feat[base + 5] = obj.size[2] / _SIZE_SCALE
    feat[base + 6] = np.sin(obj.yaw)
    feat[base + 7] = np.cos(obj.yaw)
    feat[base + 8] = obj.range_xy / _POS_SCALE
    feat[base + 9] = np.sin(obj.bearing)
    feat[base + 10] = np.cos(obj.bearing)
    feat[base + 11] = obj.score
    return feat


def encode_scene(scene: Scene, max_objects: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(features, mask)`` of shapes ``(max_objects, D)`` and ``(max_objects,)``.

    Objects beyond ``max_objects`` are dropped by increasing range, i.e. the
    farthest are discarded first. The number of truncated scenes is reported by
    :func:`truncation_rate` so this does not become an invisible bias.
    """
    objs = sorted(scene.objects, key=lambda o: o.range_xy)[:max_objects]
    feats = np.zeros((max_objects, OBJECT_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(max_objects, dtype=np.float32)
    for i, o in enumerate(objs):
        feats[i] = encode_object(o)
        mask[i] = 1.0
    return feats, mask


def truncation_rate(scenes: Iterable[Scene], max_objects: int = 64) -> float:
    scenes = list(scenes)
    if not scenes:
        return 0.0
    return sum(1 for s in scenes if len(s.objects) > max_objects) / len(scenes)


class HashingQuestionEncoder:
    """Fixed-size bag-of-words encoder: no pretrained weights, no vocabulary file.

    Unigrams and bigrams are hashed into ``dim`` buckets. This is intentionally
    the cheapest possible text encoder: the experiment is about *where the
    computation happens*, not about language modelling, and any gain that
    required a large text encoder would confound the comparison.

    The bucket index uses CRC32 rather than the built-in ``hash``: string
    hashing in CPython is salted per process, so ``hash`` would silently
    produce a different encoder on every run and make results irreproducible.
    """

    def __init__(self, dim: int = 256, use_bigrams: bool = True) -> None:
        self.dim = dim
        self.use_bigrams = use_bigrams

    def _tokens(self, question: str) -> List[str]:
        words = re.findall(r"[a-z0-9]+", question.lower())
        if not self.use_bigrams:
            return words
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    @staticmethod
    def _bucket(token: str, dim: int) -> int:
        return zlib.crc32(token.encode("utf-8")) % dim

    def encode(self, question: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(question):
            vec[self._bucket(tok, self.dim)] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def encode_batch(self, questions: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode(q) for q in questions]) if questions \
            else np.zeros((0, self.dim), dtype=np.float32)


@dataclass
class EncodedBatch:
    """Everything a learned baseline needs for one batch."""

    question: np.ndarray        # (B, Dq)
    objects: np.ndarray         # (B, N, Do)
    mask: np.ndarray            # (B, N)
    target: np.ndarray          # (B,) answer indices
    target_count: np.ndarray    # (B,) integer counts, -1 where not a count
    items: List[QAItem] = field(default_factory=list)


def encode_batch(
    items: Sequence[QAItem],
    scenes: Dict[str, Scene],
    vocab: AnswerVocabulary,
    question_encoder: HashingQuestionEncoder,
    max_objects: int = 64,
) -> EncodedBatch:
    q = question_encoder.encode_batch([it.question for it in items])
    feats, masks = [], []
    for it in items:
        scene = scenes.get(it.sample_token)
        if scene is None:
            scene = Scene(sample_token=it.sample_token, objects=[])
        f, m = encode_scene(scene, max_objects)
        feats.append(f)
        masks.append(m)
    targets = np.array([vocab.encode(it.answer) for it in items], dtype=np.int64)
    counts = np.array(
        [int(it.answer) if it.answer_type == "count" else -1 for it in items],
        dtype=np.int64,
    )
    return EncodedBatch(
        question=q,
        objects=np.stack(feats) if feats else np.zeros((0, max_objects, OBJECT_FEATURE_DIM), np.float32),
        mask=np.stack(masks) if masks else np.zeros((0, max_objects), np.float32),
        target=targets,
        target_count=counts,
        items=list(items),
    )
