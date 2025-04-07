from transformers import BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from data_loader import load_and_preprocess
import config

tokenized_datasets, tokenizer = load_and_preprocess()
model = BartForConditionalGeneration.from_pretrained(config.MODEL_NAME)

training_args = TrainingArguments(
    output_dir="./models/fine-tuned-bart-arxiv",
    evaluation_strategy="epoch",
    learning_rate=config.LEARNING_RATE,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    weight_decay=0.01,
    num_train_epochs=config.NUM_EPOCHS,
    save_total_limit=2,
    fp16=True,
    predict_with_generate=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized_datasets["validation"].select(range(500)),
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()
model.save_pretrained("./models/fine-tuned-bart-arxiv")
tokenizer.save_pretrained("./models/fine-tuned-bart-arxiv")
