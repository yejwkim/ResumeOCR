Hyperparameters Priority
1. Learning rate (and scheduler/warmup)
2. Batch size (or effective batch size via gradient accumulation)
3. Weight decay
4. Warmup steps (if using a scheduler)
5. Layer‐wise learning-rate ratios (head vs. body)
6. Other: gradient clipping, fp16 vs. fp32, early-stop patience, etc.

Hyperparameter Tuning Log
1. Baseline (F1: 0.0013)
2. Fine-tune head + last two layers with a single LR (F1: 0.0315)
3. Two-stage optimizer with layer-wise LRs (F1: 0.2979)