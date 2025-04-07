from transformers import BartForConditionalGeneration, BartTokenizer
import torch

model = BartForConditionalGeneration.from_pretrained("./models/fine-tuned-bart-arxiv")
tokenizer = BartTokenizer.from_pretrained("./models/fine-tuned-bart-arxiv")

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
