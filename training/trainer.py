import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Union, Callable
from models.base_model import BaseModel
from training.callbacks import BaseCallback, ModelCheckpoint, EarlyStopping
from utils.logger import Logger
import random
from tqdm import tqdm
from evaluation.metrics import (
    compute_bleu_scores,
    compute_bert_score,
    compute_classification_accuracy,
    compute_bev_miou,
    compute_top1_accuracy,
)

class Trainer:
    def __init__(
        self,
        model: BaseModel,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str,
        config: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.logger = logger

        self.model.to(self.device)

        self.epoch = 0
        self.best_score = float('inf')
        self.metrics = {}

        self.history = {
            "train_loss": [],
            "train_token_accuracy": [],
            "train_perplexity": [],
            "val_loss": [],
            "val_token_accuracy": [],
            "val_perplexity": [],
            "learning_rate": [],
            "epoch_time": []
        }

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        running_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        num_samples = 0

        for batch_idx, batch in tqdm(enumerate(train_loader), leave=False, desc="Training"):
            new_prompts = []
            new_answers = []
            for prompt, target in zip(batch["prompt"], batch["answer"]):
                # If target is empty, set to "none"
                if len(target) > 0:
                    random_index = random.randint(0, len(target) - 1)
                    new_prompts.append(f"{prompt} : {target[:random_index]}")
                    new_answers.append(target[random_index])
                else:
                    new_prompts.append(f"{prompt} : ")
                    new_answers.append("none")
            
            batch_loss = 0.0
            batch_valid_tokens = 0

            for i in range(len(batch["points"])):
                sample = batch["points"][i].to(self.device)
                prompt_i = new_prompts[i]
                answer_i = new_answers[i]
                
                sample = sample.unsqueeze(0)
                loss, valid_tokens = self.model(sample, [prompt_i], [answer_i], self.criterion)
                batch_loss += loss
                batch_valid_tokens += valid_tokens
                num_samples += 1

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += batch_loss.item()
            correct_tokens += batch_valid_tokens
            total_tokens += len(batch["points"])

            if batch_idx % 20 == 0 and self.logger:
                self.logger.log({
                    "training_loss_step": batch_loss.item() / len(batch["points"]),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })

        train_loss = running_loss / len(train_loader)
        token_accuracy = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(train_loss)).item()

        metrics = {
            "train_loss": train_loss,
            "train_token_accuracy": token_accuracy,
            "train_perplexity": perplexity,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }
        return metrics


    def validate(self, val_loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        total_tokens = 0
        correct_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_loader), leave=False, desc="Validation"):
                new_prompts = []
                new_answers = []
                for prompt, target in zip(batch["prompt"], batch["answer"]):
                    if len(target) > 0:
                        random_index = random.randint(0, len(target) - 1)
                        new_prompts.append(f"{prompt} : {target[:random_index]}")
                        new_answers.append(target[random_index])
                    else:
                        new_prompts.append(f"{prompt} : ")
                        new_answers.append("none")
                
                batch_loss = 0.0
                batch_valid_tokens = 0

                for i in range(len(batch["points"])):
                    sample = batch["points"][i].to(self.device)
                    prompt_i = new_prompts[i]
                    answer_i = new_answers[i]
                    sample = sample.unsqueeze(0)
                    loss, valid_tokens = self.model(sample, [prompt_i], [answer_i], self.criterion)
                    batch_loss += loss
                    batch_valid_tokens += valid_tokens
                    total_tokens += 1

                running_loss += batch_loss.item()
                correct_tokens += batch_valid_tokens

        val_loss = running_loss / len(val_loader)
        token_accuracy = 100.0 * correct_tokens / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(val_loss)).item()

        metrics = {
            "val_loss": val_loss,
            "val_token_accuracy": token_accuracy,
            "val_perplexity": perplexity,
        }
        return metrics

    
    def evaluate_captioning(self, eval_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_references = []
        all_predictions = []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating Captioning"):
                for i, sample in enumerate(batch["points"]):
                    sample = sample.to(self.device).unsqueeze(0)
                    prompt_i = batch["prompt"][i]
                    generated = self.model.generate(sample, [prompt_i])
                    all_predictions.append(generated[0])
                all_references.extend(batch["answer"])
        bleu_scores = compute_bleu_scores(all_references, all_predictions)
        bert = compute_bert_score(all_references, all_predictions)
        metrics = {**bleu_scores, "bert_score": bert}
        return metrics


    
    ### Not working yet
    """ def evaluate_grounding(self, eval_loader: DataLoader) -> Dict[str, float]:
    self.model.eval()
    all_pred_logits = []
    all_gt_labels = []
    all_pred_boxes = []
    all_gt_boxes = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating Grounding"):
            for i, sample in enumerate(batch["points"]):
                sample = sample.to(self.device).unsqueeze(0)
                prompt_i = batch["prompt"][i]
                outputs = self.model(sample, [prompt_i])
                all_pred_logits.append(outputs["logits"])
                all_gt_labels.append(batch["class_label"][i].to(self.device))
                all_pred_boxes.extend(outputs["boxes"])
                all_gt_boxes.extend(batch["bev_box"][i])
    pred_logits = torch.cat(all_pred_logits, dim=0)
    gt_labels = torch.cat(all_gt_labels, dim=0)
    classification_acc = compute_classification_accuracy(pred_logits, gt_labels)
    bev_miou = compute_bev_miou(all_pred_boxes, all_gt_boxes)
    return {"classification_accuracy": classification_acc, "bev_miou": bev_miou}


    def evaluate_highlevel(self, eval_loader: DataLoader) -> Dict[str, float]:
    self.model.eval()
    all_predictions = []
    all_gt_answers = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating High-Level Instructions"):
            for i, sample in enumerate(batch["points"]):
                sample = sample.to(self.device).unsqueeze(0)
                prompt_i = batch["prompt"][i]
                generated = self.model.generate(sample, [prompt_i])
                all_predictions.append(generated[0])
            all_gt_answers.extend(batch["answer"])
    top1_acc = compute_top1_accuracy(all_predictions, all_gt_answers)
    return {"top1_accuracy": top1_acc} """



    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ):
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.epoch = epoch
            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.metrics = {**train_metrics, **val_metrics}
            epoch_time = time.time() - start_time
            self.metrics["epoch_time"] = epoch_time

            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_token_accuracy"].append(train_metrics["train_token_accuracy"])
            self.history["train_perplexity"].append(train_metrics["train_perplexity"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_token_accuracy"].append(val_metrics["val_token_accuracy"])
            self.history["val_perplexity"].append(val_metrics["val_perplexity"])
            self.history["learning_rate"].append(train_metrics["learning_rate"])
            self.history["epoch_time"].append(epoch_time)

            if self.logger:
                self.logger.log(self.metrics, epoch=epoch)

            print(f"Epoch: {epoch+1}/{epochs} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Train Token Acc: {train_metrics['train_token_accuracy']:.2f}% | Val Token Acc: {val_metrics['val_token_accuracy']:.2f}% | "
                  f"Train PPL: {train_metrics['train_perplexity']:.2f} | Val PPL: {val_metrics['val_perplexity']:.2f} | "
                  f"Time: {epoch_time:.2f}s")

            if self.scheduler:
                self.scheduler.step()

            for callback in self.callbacks:
                callback.on_epoch_end(self)

            if any(getattr(callback, 'stop_training', False) for callback in self.callbacks):
                print(f"Training stopped early at epoch {epoch+1}")
                break

        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.metrics