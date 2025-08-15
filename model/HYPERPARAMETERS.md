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
4. AMP Clipping (F1: 0.3812)
5. Longer Epoch, Faster Unfreeze, Re-warmup (F1: 0.4431)
6. Change in Encoder LR (3e-5) + Head LR (5e-4) (F1: 0.4828)
![Alt text](tuning1.png?raw=true "Sweep #1")
7. Sweep performed over different combinations of Encoder LR and re-warm length, but no better results found. (F1: 0.4828)
![Alt text](tuning2.png?raw=true "Sweep #2")
8. Change in Layer Decay (0.7) (F1: 0.4885)
![Alt text](tuning3.png?raw=true "Sweep #3")