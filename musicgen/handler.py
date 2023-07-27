from typing import Dict, List, Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch

class EndpointHandler:
    def __init__(self, path="facebook/musicgen-small"):
        # load model and processor from path
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = MusicgenForConditionalGeneration.from_pretrained(path).to("cuda")

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
            outputs = self.model.generate(**inputs, max_new_tokens=256, **parameters)
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=256)

        # postprocess the prediction
        prediction = outputs[0].cpu().numpy()

        return [{"generated_audio": prediction}]