"""Learned baselines, all operating on the same fixed object lists.

Four systems are compared in the report. Three of them live here:

``QuestionOnlyModel``
    Sees the question and nothing else. Its score is the language prior: the
    accuracy reachable without looking at the scene at all. Any system that
    does not clearly beat it has not been shown to use the 3D input.

``DeepSetsVQA``
    Question + a permutation-invariant pooling of the object list, then a
    classifier over the answer vocabulary. This is the "predict the answer
    directly" condition, i.e. a miniature of what the LiDAR-LLM head does,
    minus the language model.

``NeuralCounterVQA``
    Same backbone, but with an explicit per-object relevance head whose sum is
    trained to equal the count. The inductive bias is "an answer of N comes
    from N objects", without hard-coding which ones.

The fourth system, ``SymbolicSpatialQA``, is in ``programs.py`` and has no
parameters.

Everything here is deliberately small enough to train on a CPU in minutes:
the point of the study is that the comparison is affordable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import (
    AnswerVocabulary,
    EncodedBatch,
    HashingQuestionEncoder,
    OBJECT_FEATURE_DIM,
    QAItem,
    encode_batch,
)
from .objects import Scene


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the object axis, ignoring padding. ``x`` is ``(B, N, D)``."""
    m = mask.unsqueeze(-1)
    total = (x * m).sum(dim=1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return total / denom


def masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask.unsqueeze(-1)).sum(dim=1)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class QuestionOnlyModel(nn.Module):
    """Answer from the question alone -- the language-prior control."""

    def __init__(self, question_dim: int, num_answers: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(question_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_answers),
        )

    def forward(self, question: torch.Tensor, objects: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"logits": self.net(question)}


class DeepSetsVQA(nn.Module):
    """Question + pooled object set -> answer class.

    The question vector is concatenated to every object before pooling, so the
    per-object encoder can already be question-conditional. Both sum and mean
    pooling are kept: sum carries cardinality (which counting needs) while mean
    is scale-stable, and dropping either one measurably hurts.
    """

    def __init__(self, question_dim: int, num_answers: int,
                 object_dim: int = OBJECT_FEATURE_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.object_encoder = nn.Sequential(
            nn.Linear(object_dim + question_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + question_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_answers),
        )

    def encode_objects(self, question: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        n = objects.shape[1]
        q = question.unsqueeze(1).expand(-1, n, -1)
        return self.object_encoder(torch.cat([objects, q], dim=-1))

    def forward(self, question: torch.Tensor, objects: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encode_objects(question, objects)
        pooled = torch.cat([masked_sum(h, mask) / 10.0, masked_mean(h, mask)], dim=-1)
        return {"logits": self.head(torch.cat([pooled, question], dim=-1))}


class NeuralCounterVQA(DeepSetsVQA):
    """Deep-sets backbone plus an explicit, differentiable counter.

    ``relevance`` is a per-object probability that the object is one of the
    objects the question is asking about. Their sum is the predicted count.
    The classification head is kept for binary questions, and the two losses
    are combined in :class:`Trainer`.
    """

    def __init__(self, question_dim: int, num_answers: int,
                 object_dim: int = OBJECT_FEATURE_DIM, hidden: int = 128) -> None:
        super().__init__(question_dim, num_answers, object_dim, hidden)
        self.relevance = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, question: torch.Tensor, objects: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encode_objects(question, objects)
        rel = torch.sigmoid(self.relevance(h)).squeeze(-1) * mask   # (B, N)
        count = rel.sum(dim=1)                                      # (B,)
        pooled = torch.cat([masked_sum(h, mask) / 10.0, masked_mean(h, mask)], dim=-1)
        logits = self.head(torch.cat([pooled, question], dim=-1))
        return {"logits": logits, "count": count, "relevance": rel}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    count_loss_weight: float = 1.0
    max_objects: int = 64
    seed: int = 0
    device: str = "cpu"
    patience: int = 8          # early stopping on validation accuracy
    verbose: bool = False


@dataclass
class TrainReport:
    """What a training run cost and how it went -- both are reported."""

    epochs_run: int = 0
    best_val_accuracy: float = 0.0
    train_seconds: float = 0.0
    num_parameters: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


class LearnedSystem:
    """Wraps a model so that all four systems share one ``predict`` interface."""

    def __init__(
        self,
        model: nn.Module,
        vocab: AnswerVocabulary,
        question_encoder: HashingQuestionEncoder,
        config: TrainConfig,
        name: str = "learned",
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.question_encoder = question_encoder
        self.config = config
        self.name = name
        self.report = TrainReport(
            num_parameters=sum(p.numel() for p in model.parameters())
        )

    # -- training --------------------------------------------------------- #

    def fit(
        self,
        train_items: Sequence[QAItem],
        scenes: Dict[str, Scene],
        val_items: Optional[Sequence[QAItem]] = None,
    ) -> TrainReport:
        torch.manual_seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)
        device = torch.device(self.config.device)
        self.model.to(device)

        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        items = list(train_items)
        best_state, best_acc, since_best = None, -1.0, 0
        start = time.perf_counter()

        for epoch in range(self.config.epochs):
            self.model.train()
            order = rng.permutation(len(items))
            epoch_loss, seen = 0.0, 0

            for i in range(0, len(order), self.config.batch_size):
                chunk = [items[j] for j in order[i:i + self.config.batch_size]]
                batch = encode_batch(chunk, scenes, self.vocab,
                                     self.question_encoder, self.config.max_objects)
                loss = self._loss(batch, device)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += float(loss.item()) * len(chunk)
                seen += len(chunk)

            entry = {"epoch": float(epoch), "train_loss": epoch_loss / max(seen, 1)}

            if val_items:
                acc = self.accuracy(val_items, scenes)
                entry["val_accuracy"] = acc
                if acc > best_acc:
                    best_acc, since_best = acc, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    since_best += 1

            self.report.history.append(entry)
            self.report.epochs_run = epoch + 1
            if self.config.verbose:
                print(f"[{self.name}] {entry}")
            if val_items and since_best >= self.config.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.report.best_val_accuracy = max(best_acc, 0.0)
        self.report.train_seconds = time.perf_counter() - start
        return self.report

    def _loss(self, batch: EncodedBatch, device: torch.device) -> torch.Tensor:
        question = torch.from_numpy(batch.question).to(device)
        objects = torch.from_numpy(batch.objects).to(device)
        mask = torch.from_numpy(batch.mask).to(device)
        target = torch.from_numpy(batch.target).to(device)

        out = self.model(question, objects, mask)

        # Items whose answer is outside the vocabulary (index -1) are excluded
        # from the classification loss but still counted as errors at test time.
        valid = target >= 0
        if valid.any():
            loss = F.cross_entropy(out["logits"][valid], target[valid])
        else:
            loss = out["logits"].sum() * 0.0

        if "count" in out and self.config.count_loss_weight > 0:
            target_count = torch.from_numpy(batch.target_count).to(device).float()
            is_count = target_count >= 0
            if is_count.any():
                loss = loss + self.config.count_loss_weight * F.l1_loss(
                    out["count"][is_count], target_count[is_count]
                )
        return loss

    # -- inference -------------------------------------------------------- #

    @torch.no_grad()
    def predict(
        self,
        items: Sequence[QAItem],
        scenes: Dict[str, Scene],
        batch_size: int = 64,
    ) -> List[str]:
        self.model.eval()
        device = torch.device(self.config.device)
        predictions: List[str] = []

        for i in range(0, len(items), batch_size):
            chunk = list(items[i:i + batch_size])
            batch = encode_batch(chunk, scenes, self.vocab,
                                 self.question_encoder, self.config.max_objects)
            out = self.model(
                torch.from_numpy(batch.question).to(device),
                torch.from_numpy(batch.objects).to(device),
                torch.from_numpy(batch.mask).to(device),
            )
            classes = out["logits"].argmax(dim=-1).cpu().numpy()

            for k, item in enumerate(chunk):
                if "count" in out and item.answer_type == "count":
                    # For counting questions the explicit counter is the output;
                    # the classifier head is only used for the other types.
                    value = int(round(float(out["count"][k].item())))
                    predictions.append(str(max(value, 0)))
                else:
                    predictions.append(self.vocab.decode(int(classes[k])))
        return predictions

    def accuracy(self, items: Sequence[QAItem], scenes: Dict[str, Scene]) -> float:
        if not items:
            return 0.0
        preds = self.predict(items, scenes)
        hits = sum(1 for p, it in zip(preds, items) if p == it.answer)
        return hits / len(items)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_system(
    kind: str,
    vocab: AnswerVocabulary,
    question_encoder: HashingQuestionEncoder,
    config: Optional[TrainConfig] = None,
) -> LearnedSystem:
    """``kind`` is ``"question_only"``, ``"deepsets"`` or ``"counter"``."""
    config = config or TrainConfig()
    qdim = question_encoder.dim
    if kind == "question_only":
        model: nn.Module = QuestionOnlyModel(qdim, len(vocab))
    elif kind == "deepsets":
        model = DeepSetsVQA(qdim, len(vocab))
    elif kind == "counter":
        model = NeuralCounterVQA(qdim, len(vocab))
    else:
        raise KeyError(f"unknown system {kind!r}")
    return LearnedSystem(model, vocab, question_encoder, config, name=kind)
