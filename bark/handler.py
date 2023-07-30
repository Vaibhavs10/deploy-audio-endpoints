from typing import Dict, List, Any
from transformers import AutoProcessor, BarkModel

import torch

class EndpointHandler:
    def __init__(self, path="suno/bark"):
        # load model and processor from path
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = BarkModel.from_pretrained(path, torch_dtype=torch.float16).to("cuda")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Args:
            data (:dict:):
                The payload with the text prompt and generation parameters.
        """
        # process input
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # preprocess
        inputs = self.processor(
            text=[inputs],
            padding=True,
            return_tensors="pt",).to("cuda")

        # pass inputs with all kwargs in data
        if parameters is not None:
            outputs = self.model.generate(**inputs, **parameters)
        else:
            outputs = self.model.generate(**inputs,)

        # postprocess the prediction
        prediction = outputs[0].cpu().numpy()

        return [{"generated_audio": prediction}]