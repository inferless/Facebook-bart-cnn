from datasets import load_dataset
from transformers import BartTokenizer
import config

def load_and_preprocess():
    dataset = load_dataset("scientific_papers", "arxiv", trust_remote_code=True)  # <-- Added this!
    tokenizer = BartTokenizer.from_pretrained(config.MODEL_NAME)

    def preprocess(example):
        inputs = tokenizer(example["article"], max_length=config.MAX_SOURCE_LENGTH, padding="max_length", truncation=True)
        targets = tokenizer(example["abstract"], max_length=config.MAX_TARGET_LENGTH, padding="max_length", truncation=True)
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["article", "abstract", "section_names"])
    return tokenized, tokenizer
