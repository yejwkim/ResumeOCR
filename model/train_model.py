import json
import torch
import re
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoProcessor, AutoModelForTokenClassification, get_scheduler, DataCollatorForTokenClassification
from datasets import load_dataset
from PIL import Image
import numpy as np
import random
import evaluate
from tqdm.auto import tqdm
import wandb
from huggingface_hub import login
import os

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
        "epochs": 30,
        "batch_size": 2,
        "head_lr": 5e-4,
        "encoder_lr": 1e-5,
        "layer_decay": 0.8,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "model": "microsoft/layoutlmv3-base",
        "head_only_epochs": 1,
        "enc_rewarm_steps": 200,
        "seed": seed
    }
)
config = wandb.config
epochs = int(config.epochs)
batch_size = int(config.batch_size)
head_lr = float(config.head_lr)
encoder_lr = float(config.encoder_lr)
layer_decay = float(config.layer_decay)
weight_decay = float(config.weight_decay)
warmup_ratio = float(config.warmup_ratio)
head_only_epochs = int(config.head_only_epochs)
enc_rewarm_steps = int(getattr(config, "enc_rewarm_steps", 200))

# For Google Colab (Change directory if ran local)
CKPT_DIR = os.getenv("CKPT_DIR", "/content/ckpts")
os.makedirs(CKPT_DIR, exist_ok=True)
run_id = wandb.run.id if wandb.run and hasattr(wandb.run, "id") else "unknown"
best_ckpt_path = os.path.join(CKPT_DIR, f"best_{run_id}.pt")
last_best_path = None

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

# Stage 1 Freezing (Head Only)
for _, p in model.named_parameters():
    p.requires_grad = False
for name, p in model.named_parameters(): # Only classifier head enabled
    if "classifier" in name:
        p.requires_grad = True

# Stage 2 Helper
layer_pat = re.compile(r"(?:^|\.)(encoder)\.layer\.(\d+)\.")
def enable_last_two_layers():
    for name, p in model.named_parameters():
        m = layer_pat.search(name)
        if m:
            lid = int(m.group(2))
            if lid >= 10:
                p.requires_grad = True

# Optimizer Parameter Groups
def is_no_decay(n):
    return n.endswith(".bias") or "LayerNorm.weight" in n or "layernorm.weight" in n

head_decay: list[torch.nn.Parameter] = []
head_no_decay: list[torch.nn.Parameter] = []
enc10_decay: list[torch.nn.Parameter] = []
enc10_no_decay: list[torch.nn.Parameter] = []
enc11_decay: list[torch.nn.Parameter] = []
enc11_no_decay: list[torch.nn.Parameter] = []

for name, p in model.named_parameters():
    if "classifier" in name:
        (head_no_decay if is_no_decay(name) else head_decay).append(p)
        continue
    m = layer_pat.search(name)
    if not m:
        continue
    layer_id = int(m.group(2))
    if layer_id == 10:
        (enc10_no_decay if is_no_decay(name) else enc10_decay).append(p)
    elif layer_id == 11:
        (enc11_no_decay if is_no_decay(name) else enc11_decay).append(p)

enc_l11_lr = encoder_lr
enc_l10_lr = encoder_lr * layer_decay

optimizer = torch.optim.AdamW(
    [
        {"name":"head-decay", "params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
        {"name":"head-nodecay", "params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        {"name":"enc-l10-decay", "params": enc10_decay, "lr": enc_l10_lr, "weight_decay": weight_decay},
        {"name":"enc-l10-nodecay", "params": enc10_no_decay, "lr": enc_l10_lr, "weight_decay": 0.0},
        {"name":"enc-l11-decay", "params": enc11_decay, "lr": enc_l11_lr, "weight_decay": weight_decay},
        {"name":"enc-l11-nodecay", "params": enc11_no_decay, "lr": enc_l11_lr, "weight_decay": 0.0}
    ]
)

wandb.watch(model, log="all", log_freq=50)
print("HuggingFace & optimizer setup complete.")

# Load Dataset
raw = load_dataset("json", data_files={"train": "data/hf_dataset.jsonl"}, split="train")
split = raw.train_test_split(test_size=0.1, seed=seed)
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
# token_collator = DataCollatorForTokenClassification(processor.tokenizer, padding="longest", return_tensors="pt")
# def multimodal_collate_fn(examples):
#     token_batch = token_collator([{k: v for k, v in ex.items() if k in ("input_ids","attention_mask","labels")} for ex in examples])
#     pixel_list = [torch.tensor(ex["pixel_values"]) for ex in examples]
#     token_batch["pixel_values"] = torch.stack(pixel_list, dim=0)
#     L_max = token_batch["input_ids"].size(1)
#     padded_bboxes = []
#     for ex in examples:
#         b = torch.tensor(ex["bbox"], dtype=torch.long)  # shape (L, 4)
#         if b.ndim == 1:  # just in case
#             b = b.view(-1, 4)
#         pad_len = L_max - b.size(0)
#         if pad_len > 0:
#             pad_boxes = torch.zeros(pad_len, 4, dtype=b.dtype)
#             b = torch.cat([b, pad_boxes], dim=0)
#         padded_bboxes.append(b)
#     token_batch["bbox"] = torch.stack(padded_bboxes, dim=0)
#     return token_batch
# train_loader = DataLoader(encoded_train, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate_fn, num_workers=2, pin_memory=True)
# val_loader = DataLoader(encoded_val, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate_fn, num_workers=2, pin_memory=True)
train_loader = DataLoader(encoded_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(encoded_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoader complete")

# Scheduler
num_training_steps = epochs * len(train_loader)
warmup_steps = int(warmup_ratio * num_training_steps)
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

# Metric & Early stopping
metric = evaluate.load("seqeval")
best_f1 = 0.0
patience = 5
no_improve = 0
print("Pre-training setup complete")

# Training Loop
scaler = GradScaler(enabled=(device.type == "cuda"))
def log_group_lrs(step: int):
    lr_dict = {g.get("name", f"pg{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)}
    wandb.log({**{f"lr/{k}": v for k, v in lr_dict.items()}}, step=step)
def print_trainable(prefix=""):
    hot = [n for n, p in model.named_parameters() if p.requires_grad]
    # print(f"{prefix} Trainable params: {len(hot)}")
    if hot:
        print("  ", "\n  ".join(hot[:10]), "..." if len(hot) > 10 else "")
try:
    print_trainable(prefix="Stage 1 (head-only):") # Stage 1
    for epoch in range(1, epochs + 1):
        # Stage Switch
        if epoch == int(head_only_epochs) + 1:
            enable_last_two_layers()
            # print_trainable(prefix="Stage 2 (head + last2):")
            rewarm_start = (epoch - 1) * len(train_loader)
            rewarm_end   = rewarm_start + int(enc_rewarm_steps)
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
            scaler.unscale_(optimizer) # AMP-safe Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # Encoder rewarm
            if 'rewarm_start' in locals():
                if global_step <= rewarm_end: # Linear Ramp
                    t = (global_step - rewarm_start) / max(1, (rewarm_end - rewarm_start))
                    t = float(max(0.0, min(1.0, t)))
                    for g in optimizer.param_groups:
                        name = g.get("name","")
                        if name.startswith("enc-"):
                            base = enc_l11_lr if "l11" in name else enc_l10_lr
                            g["lr"] = base * t
                elif global_step == rewarm_end + 1: # Snap to base once ramp is over
                    for g in optimizer.param_groups:
                        name = g.get("name","")
                        if name.startswith("enc-"):
                            g["lr"] = enc_l11_lr if "l11" in name else enc_l10_lr

            total_loss += loss.item()
            train_iter.set_postfix(loss=total_loss/step)
            lr = scheduler.get_last_lr()[0]
            log_group_lrs(global_step)
            wandb.log({"train_loss_step": loss.item()}, step=global_step)
            
            logits = outputs.logits.detach().float().cpu().numpy()
            label_ids = batch["labels"].detach().cpu().numpy()
            preds_batch = np.argmax(logits, axis=2)
            for pred_seq, label_seq in zip(preds_batch, label_ids):
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    train_preds.append(p)
                    train_labels.append(l)
        
        avg_train_loss = total_loss / max(1, len(train_loader))
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

        avg_val_loss = val_loss / max(1, val_steps)
        results = metric.compute(predictions=all_preds, references=all_labels, zero_division=0)
        f1 = results.get("overall_f1", 0.0)
        print(f"Epoch {epoch} Validation → Loss: {avg_val_loss:.4f}, F1: {f1:.4f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss,
                   "train_accuracy": train_acc, "val_precision": results.get("overall_precision", 0.0),
                   "val_recall": results.get("overall_recall", 0.0),
                   "val_f1": f1, "epoch": epoch}, step=global_step)

        # Early stopping check
        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            try:
                payload = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": dict(config),
                        "epoch": epoch,
                        "best_f1": best_f1,
                }
                torch.save(payload, best_ckpt_path)
                
                if last_best_path and last_best_path != best_ckpt_path and os.path.exists(last_best_path):
                    try:
                        os.remove(last_best_path)
                    except OSError:
                        print(f"{last_best_path} removal failed.")
                        pass
                last_best_path = best_ckpt_path
                
                run_id = wandb.run.id if wandb.run and hasattr(wandb.run, "id") else "unknown"
                artifact = wandb.Artifact(f"layoutlmv3-resume-{run_id}", type="model", 
                                        metadata={"f1": best_f1, "epoch": epoch})
                artifact.add_file(best_ckpt_path)
                wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])
                print(f"New best F1, saved to {best_ckpt_path}.")
            except Exception as e:
                print(f"Checkpoint save/upload error: {e}. Continuing training.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No improvement in {patience} epochs, stopping.")
                break
finally:
    wandb.finish()
    print("Training complete.")
