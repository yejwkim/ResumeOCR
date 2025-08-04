import json
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoProcessor, AutoModelForTokenClassification, get_scheduler
from datasets import load_dataset
from PIL import Image
import numpy as np
import random
import evaluate
from tqdm.auto import tqdm
import wandb
from huggingface_hub import login

print("Import complete.")

# Device & Seed Setup
login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)

# W&B Setup
wandb.init(
    project="resume-ocr",
    entity="yejwkim-the-university-of-texas-at-austin",
    config={
        "epochs": 15,
        "batch_size": 2,
        "head_lr": 5e-4,
        "encoder_lr": 1e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "model": "microsoft/layoutlmv3-base",
        "seed": seed
    }
)
config = wandb.config
epochs = config.epochs

# Label Extraction & Mapping
all_label_names = set()
with open("data/hf_dataset.jsonl", "r") as f:
    for line in f:
        ex = json.loads(line)
        all_label_names.update(ex["labels"])
label_list = sorted(all_label_names)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
print("Label setup complete.")

# Processor & Model Initialization
processor = AutoProcessor.from_pretrained(config.model, apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(
    config.model,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# Freeze
for name, param in model.named_parameters():
    if "classifier" in name:
        param.requires_grad = True
    elif any(f"encoder.layer.{i}" in name for i in [10, 11]):
        param.requires_grad = True
    else:
        param.requires_grad = False
wandb.watch(model, log="all", log_freq=50)
head_params, enc2_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "classifier" in name:
        head_params.append(p)
    else:
        enc2_params.append(p)
print("HuggingFace setup complete.")

# Load Dataset
raw = load_dataset("json", data_files={"train": "data/hf_dataset.jsonl"}, split="train")
split = raw.train_test_split(test_size=0.1, seed=config.seed)
train_raw = split["train"]
val_raw = split["test"]
print("Dataset loading complete.")

# Preprocessing
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

encoded_train = train_raw.map(preprocess, remove_columns=train_raw.column_names, num_proc=4)
encoded_val = val_raw.map(preprocess, remove_columns=val_raw.column_names, num_proc=4)
encoded_train.set_format("torch")
encoded_val.set_format("torch")
print("Dataset preprocessing complete.")

# DataLoader
train_loader = DataLoader(encoded_train, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(encoded_val, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoader complete")

# Optimizer & Scheduler
optimizer = torch.optim.AdamW([
    {"params": head_params, "lr": config.head_lr},
    {"params": enc2_params,  "lr": config.encoder_lr},
    ], weight_decay=config.weight_decay)
num_training_steps = epochs * len(train_loader)
warmup_steps = int(config.warmup_ratio * num_training_steps)
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

# Metric
metric = evaluate.load("seqeval")

# Early stopping
best_f1 = 0.0
patience = 5
no_improve = 0
print("Pre-training setup complete")

# Training Loop
scaler = GradScaler()
try:
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        train_preds, train_labels = [], []
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch} ▶ Training", leave=False)
        for step, batch in enumerate(train_iter, 1):
            global_step = (epoch - 1) * len(train_loader) + step
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type=str(device)):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_loss += loss.item()
            train_iter.set_postfix(loss=total_loss/step)
            lr = scheduler.get_last_lr()[0]
            wandb.log({"lr": lr, "train_loss_step": loss.item()}, step=global_step)
            
            logits = outputs.logits.detach().float().cpu().numpy()
            label_ids = batch["labels"].detach().cpu().numpy()
            preds_batch = np.argmax(logits, axis=2)
            for pred_seq, label_seq in zip(preds_batch, label_ids):
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    train_preds.append(p)
                    train_labels.append(l)
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = np.mean([p == l for p, l in zip(train_preds, train_labels)])
        print(f"Epoch {epoch} Training → Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        val_steps = 0
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch} ▶ Validation", leave=False)
        with torch.no_grad():
            for batch in val_iter:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_steps += 1
                logits = outputs.logits.cpu().float().numpy()
                label_ids = batch["labels"].cpu().numpy()

                preds = np.argmax(logits, axis=2)
                for pred_seq, label_seq in zip(preds, label_ids):
                    true_labels = [id2label[l] for l in label_seq if l != -100]
                    true_preds  = [id2label[p] for (p, l) in zip(pred_seq, label_seq) if l != -100]
                    all_labels.append(true_labels)
                    all_preds.append(true_preds)

        avg_val_loss = val_loss / val_steps
        results = metric.compute(predictions=all_preds, references=all_labels, zero_division=0)
        f1 = results["overall_f1"]
        print(f"Epoch {epoch} Validation → Loss: {avg_val_loss:.4f}, F1: {f1:.4f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss,
                   "train_accuracy": train_acc, "val_precision": results["overall_precision"],
                   "val_recall": results["overall_recall"],
                   "val_f1": results["overall_f1"], "epoch": epoch}, step=global_step)

        # Early stopping check
        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
            run_id = wandb.run.id if wandb.run and hasattr(wandb.run, "id") else "unknown"
            artifact = wandb.Artifact(f"layoutlmv3-resume-{run_id}", type="model", 
                                    metadata={"f1": best_f1, "epoch": epoch})
            artifact.add_file("best_model.pt")
            wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
            print("New best F1, saving model.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement in {patience} epochs, stopping.")
                break
finally:
    wandb.finish()
    print("Training complete.")