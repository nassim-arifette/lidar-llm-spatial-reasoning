import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseCallback(ABC):
    @abstractmethod
    def on_train_begin(self, trainer):
        pass

    @abstractmethod
    def on_train_end(self, trainer):
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer):
        pass


class ModelCheckpoint(BaseCallback):
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: int = 1,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best = -float('inf')
            self.monitor_op = lambda x, y: x > y

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        current = trainer.metrics.get(self.monitor)
        if current is None:
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f"\nEpoch {trainer.epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model to {self.filepath}")
                self.best = current
                trainer.model.save(self.filepath)
        else:
            filepath = self.filepath.format(epoch=trainer.epoch, **trainer.metrics)
            if self.verbose > 0:
                print(f"\nEpoch {trainer.epoch+1}: saving model to {filepath}")
            trainer.model.save(filepath)


class EarlyStopping(BaseCallback):
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0,
        patience: int = 10,
        mode: str = 'min',
        verbose: int = 1,
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.stop_training = False

        if mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda x, y: x < y - min_delta
        else:
            self.best = -float('inf')
            self.monitor_op = lambda x, y: x > y + min_delta

        self.wait = 0

    def on_train_begin(self, trainer):
        self.wait = 0
        self.stop_training = False

    def on_train_end(self, trainer):
        trainer.model.save("checkpoints/" + str(trainer.epoch) + ".pth")

    def on_epoch_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        current = trainer.metrics.get(self.monitor)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f"\nEarly stopping triggered: {self.monitor} didn't improve for {self.patience} epochs")
                self.stop_training = True
