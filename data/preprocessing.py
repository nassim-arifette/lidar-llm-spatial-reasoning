import numpy as np
from typing import Dict, List, Union, Tuple, Callable
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError


class StandardScaler(BasePreprocessor):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return self

    def transform(self, data: np.ndarray):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data: np.ndarray):
        return data * self.std + self.mean


class MinMaxScaler(BasePreprocessor):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return self

    def transform(self, data: np.ndarray):
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data: np.ndarray):
        return data * (self.max - self.min) + self.min


class PreprocessingPipeline:
    def __init__(self, preprocessors: List[BasePreprocessor]):
        self.preprocessors = preprocessors

    def fit(self, data):
        for preprocessor in self.preprocessors:
            preprocessor.fit(data)
            data = preprocessor.transform(data)
        return self

    def transform(self, data):
        for preprocessor in self.preprocessors:
            data = preprocessor.transform(data)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
