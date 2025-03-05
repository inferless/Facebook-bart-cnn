import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from transformers import pipeline

class InferlessPythonModel:
    def initialize(self):
        snapshot_download(repo_id="facebook/bart-large-cnn",allow_patterns=["*.safetensors"])
        self.generator = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cuda",
        )

    def infer(self, inputs):
        prompt = inputs["text"]
        pipeline_output = self.generator(prompt)
        return {"generated_text":pipeline_output[0]["summary_text"]}

    def finalize(self):
        self.pipe = None
