import math
from typing import List, Dict
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

def compute_bleu_scores(references: List[str], candidates: List[str]) -> Dict[str, float]:
    smooth_fn = SmoothingFunction().method1
    bleu_scores = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    total = len(candidates)
    for ref, cand in zip(references, candidates):
        ref_tokens = ref.split()
        cand_tokens = cand.split()
        bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
        bleu2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
        bleu3 = sentence_bleu([ref_tokens], cand_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth_fn)
        bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
        bleu_scores["bleu1"] += bleu1
        bleu_scores["bleu2"] += bleu2
        bleu_scores["bleu3"] += bleu3
        bleu_scores["bleu4"] += bleu4

    for key in bleu_scores:
        bleu_scores[key] /= total
    return bleu_scores

def compute_bert_score(references: List[str], candidates: List[str]) -> float:

    if bert_score is None:
        raise ImportError("Please install bert-score: pip install bert-score")
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
    return F1.mean().item()

def compute_classification_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:

    pred_labels = predictions.argmax(dim=1)
    correct = (pred_labels == targets).sum().item()
    total = targets.numel()
    return 100.0 * correct / total

def compute_bev_miou(pred_boxes: List[List[float]], gt_boxes: List[List[float]]) -> float:

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
    
    if len(pred_boxes) == 0 or len(pred_boxes) != len(gt_boxes):
        return 0.0
    ious = [iou(pred, gt) for pred, gt in zip(pred_boxes, gt_boxes)]
    return sum(ious) / len(ious)

def compute_top1_accuracy(predictions: List[str], targets: List[str]) -> float:

    correct = sum(1 for pred, target in zip(predictions, targets)
                  if pred.strip().lower() == target.strip().lower())
    total = len(targets)
    return 100.0 * correct / total
