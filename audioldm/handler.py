from typing import Dict, List, Any
from diffusers import AudioLDMPipeline

import torch

class EndpointHandler:
    def __init__(self, path=""):
        # load model and processor from path
        self.pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Args:
            data (:dict:):
                The payload with the text prompt and generation parameters.
        """
        # process input
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)      

        # pass inputs with all kwargs in data
        if parameters is not None:
            outputs = self.pipe(inputs, **parameters)
        else:
            outputs = self.pipe(inputs,)

        # postprocess the prediction
        prediction = outputs[0].cpu().numpy()

        return [{"generated_audio": prediction}]