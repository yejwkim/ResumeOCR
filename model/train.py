import json
import logging
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForTokenClassification, TrainingArguments, Trainer
from PIL import Image

print("Import complete.")

labels = set()
with open("data/hf_dataset.jsonl", "r") as f:
    for line in f:
        ex = json.loads(line)
        labels.update(ex["labels"])
label_list = sorted(labels)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
print("Label setup complete.")

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)
print("HuggingFace setup complete.")

dataset = load_dataset("json", data_files={"train": "data/hf_dataset.jsonl"}, split="train")
print("Dataset loading complete.")

def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    bboxes = [[round(x * 10) for x in box] for box in example["bboxes"]]
    encoding = processor(
        image,
        text=example["words"],
        boxes=bboxes,
        word_labels=[label2id[l] for l in example["labels"]],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return {k: v.squeeze(0) for k, v in encoding.items()}

encoded_dataset = dataset.map(preprocess)
encoded_dataset.set_format("torch")
print("Dataset preprocessing complete.")

training_args = TrainingArguments(
    output_dir="./layoutlmv3-resume",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    save_steps=500,
    logging_steps=100,
    remove_unused_columns=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    processing_class=processor
)
print("Training setup complete.")

trainer.train()
print("Training complete.")
