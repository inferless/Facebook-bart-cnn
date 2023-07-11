import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0,
            use_auth_token="False",
        )

    def infer(self, text):
        pipeline_output = self.generator(text)
        return pipeline_output[0]["summary_text"]

    def finalize(self):
        self.pipe = None
