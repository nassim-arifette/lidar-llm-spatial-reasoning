import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def save(self, path: str):
        torch.save({"model_state_dict": self.state_dict()}, path)

    def load(self, path: str, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return self

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, data):
        return data
