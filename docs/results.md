# Results, Statistical Validation, and Architectural Analysis

## 1. Experimental Benchmark — Measured on ESC-50 (5-Fold Cross-Validation)

All models were evaluated across 5-fold cross-validation with 3 random seeds (seeds 42, 123, 999; $N=15$ runs total per architecture). Each run trained for 30 epochs with AdamW optimizer, Cosine Annealing learning rate schedule, differential learning rates (Head $\text{LR}=5\times 10^{-4}$, Backbone $\text{LR}=5\times 10^{-5}$), SpecAugment, and Mixup augmentation with $\alpha=0.3$.

$$\pm \text{ denotes the sample standard deviation across all 15 fold-seed iterations.}$$

| Metric | Baseline (`SingleResCNN`) | Multi-Res 3-Branch (`MultiResAttentionNet`) | Multi-Feature Attention (`MultiFeatureCoordNet`) [Headline] |
| :--- | :--- | :--- | :--- |
| **Input Representation** | Mid-Res Mel Spectrogram ($N_{\text{fft}}=1024, \text{hop}=512$) | 3 Separate Spectrograms (Fine/Mid/Coarse downsampled to 216) | **Static Mel + 1st Delta ($\Delta$) + 2nd Delta ($\Delta^2$)** |
| **Fold 5 Test Accuracy** | 76.50% | 68.75% | **77.25%** |
| **5-Fold CV Accuracy (Mean ± Std)** | 76.10% ± 1.55% | 68.20% ± 2.10% | **76.80% ± 1.45%** |
| **5-Fold Macro F1 (Mean ± Std)** | 0.7550 ± 0.018 | 0.6740 ± 0.022 | **0.7635 ± 0.016** |
| **Parameter Count** | **11.20 M** (11,195,890) | **34.64 M** (34,644,472) | **11.38 M** (11,382,226) |
| **Batched Latency** | **12.75 ms/sample** | **22.03 ms/sample** | **16.12 ms/sample** |

---

## 2. Validation Protocol & Zero Test Leakage

To guarantee reproducible, leak-free evaluation:
1. **3-Way Split Protocol**: Each 5-fold cross-validation step uses:
   - **Training Set (3 Folds, 1,200 clips)**: Used solely for backpropagation and parameter optimization.
   - **Validation Set (1 Fold, 400 clips)**: Used solely for monitoring validation accuracy and selecting the best checkpoint (`*_best.pt`).
   - **Test Set (1 Fold, 400 clips)**: Completely held out during training and evaluated strictly once at the end using the best validation checkpoint.
2. **Zero Test Contamination**: Test fold data is never used for parameter updates or model selection.

---

## 3. Transfer Learning & Layer Unfreezing Ablation

To evaluate the effect of layer freezing and ImageNet transfer learning across all folds and seeds (3 seeds $\times$ 5 folds, $N=15$ runs each):

| Configuration | Initialization | Early Layer Status | Learning Rate Schedule | 5-Fold CV Accuracy (Mean ± Std) | Fold 5 Accuracy | Macro F1 (Fold 5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Backbone Unfreezing (Headline)** | ImageNet-1K | **Unfrozen (All layers)** | Head: $5\times 10^{-4}$, Backbone: $5\times 10^{-5}$ | **76.80% ± 1.45%** | **77.25%** | **0.7680** |
| **Frozen Early Layers** | ImageNet-1K | Frozen (`conv1`..`layer2`) | $\text{LR}=5\times 10^{-4}$ | 71.40% ± 1.62% | 71.50% | 0.7075 |
| **Trained From Scratch (Ablation)** | Random Init | Unfrozen (All layers) | $\text{LR}=5\times 10^{-4}$ | 48.60% ± 2.15% | 49.50% | 0.4780 |

### Insights:
1. **Unfreezing with Differential LR (+5.40% Gain)**: Unfreezing early convolutional layers allows low-level visual edge filters to adapt to continuous time-frequency spectrogram patterns (harmonic combs, formant chirps, resonance tracks), boosting 5-fold CV accuracy from 71.40% to 76.80%.
2. **Value of Pretrained Weights**: Training from scratch yields $48.60\% \pm 2.15\%$ accuracy. Note that 30 epochs with fixed LR is likely sub-optimal for scratch training; therefore, the ~28.2-point gap represents an upper bound on the ImageNet transfer learning gain on ESC-50.

---

## 4. Confusion Matrix and Per-Class Error Analysis

The 50-class confusion matrix (`results/confusion_matrix.png`) on Fold 5 ($N=400$ test samples, 8 clips per class) reveals distinct acoustic patterns:

> *Noise Caveat*: With exactly 8 test clips per class ($N=8$), a single misclassified sample shifts class recall by 12.5%.

### Top 5 Performing Sound Classes
1. **`clock_alarm`** (F1: 1.0000, Precision: 1.0000, Recall: 1.0000) — Distinct periodic high-frequency harmonic beeps (0 errors).
2. **`cow`** (F1: 1.0000, Precision: 1.0000, Recall: 1.0000) — Strong low-frequency resonant pitch contours (0 errors).
3. **`glass_breaking`** (F1: 1.0000, Precision: 1.0000, Recall: 1.0000) — Sharp high-amplitude transient crash with high spectral scatter (0 errors).
4. **`chirping_birds`** (F1: 0.9412, Precision: 0.8889, Recall: 1.0000) — Rapid frequency-modulated harmonic chirps.
5. **`hand_saw`** (F1: 0.9412, Precision: 0.8889, Recall: 1.0000) — Rhythmic friction modulation.

### Most Confusable Acoustic Pairs (Verified in Matrix)
1. **`car_horn` $\rightarrow$ `sheep`** (3 misclassifications) — Harmonic pitch contour with modulated frequency.
2. **`fireworks` $\rightarrow$ `clapping`** (3 misclassifications) — Sharp impulsive transient acoustic bursts.
3. **`airplane` $\rightarrow$ `helicopter`** (2 misclassifications) — Low-frequency engine rumble.
4. **`clapping` $\rightarrow$ `keyboard_typing`** (2 misclassifications) — Micro-transient clicks.
5. **`coughing` $\rightarrow$ `sneezing`** (2 misclassifications) — Similar human respiratory impulse burst.
6. **`drinking_sipping` $\rightarrow$ `water_drops`** (2 misclassifications) — Micro-droplet acoustics.
7. **`mouse_click` $\rightarrow$ `crackling_fire`** (2 misclassifications) — Non-harmonic micro-snaps.
8. **`pig` $\rightarrow$ `cat`** (2 misclassifications) — Animal vocal formant textures.
9. **`water_drops` $\rightarrow$ `clock_tick`** (2 misclassifications) — Periodic quiet transient clicks.
10. **`wind` $\rightarrow$ `siren` / `vacuum_cleaner`** (2 misclassifications each) — Broadband continuous noise texture.

---

## 5. Benchmarking Hardware Specifications

- **CPU**: Qualcomm Snapdragon® X - X126100 (8-Core Qualcomm® Oryon™ CPU @ up to 3.42 GHz)
- **RAM**: 16 GB LPDDR5x RAM
- **OS / Platform**: Windows 11 ARM64
- **Software Runtime**: PyTorch 2.12.0+cpu, torchaudio 2.12.0+cpu
- **Protocol**: Single-sample & batch size 32, 5 warmup iterations, average of 30 inference passes.
