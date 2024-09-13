from transformers import pipeline


class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="cuda",
        )

    def infer(self, inputs):
        prompt = inputs["text"]
        pipeline_output = self.generator(prompt)
        return pipeline_output[0]["summary_text"]

    def finalize(self):
        self.pipe = None
