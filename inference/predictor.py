import torch
import numpy as np
from typing import Dict, Any, List, Union, Optional

class Predictor:
    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        preprocessing=None,
        postprocessing=None,
    ):
        self.model = model
        self.device = device
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

        self.model.to(self.device)
        self.model.eval()

    def predict(self, inputs):
        # If inputs["points"] is a list, process each sample individually.
        if self.preprocessing:
            inputs = self.preprocessing(inputs)

        all_generated = []
        with torch.no_grad():
            for i in range(len(inputs["points"])):
                sample = inputs["points"][i].to(self.device)
                prompt = inputs["prompt"][i]
                sample = sample.unsqueeze(0)
                generated = self.model.generate(sample, [prompt])
                all_generated.append(generated[0])
        if self.postprocessing:
            all_generated = self.postprocessing(all_generated)
        return all_generated
