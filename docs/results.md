# Results, Statistical Validation, and Architectural Analysis

## 1. Experimental Benchmark — Measured on ESC-50

All models were evaluated across 5-fold cross-validation with multiple random seeds (30 epochs per fold, AdamW optimizer, Cosine Annealing learning rate schedule, SpecAugment, and Mixup augmentation with $\alpha=0.3$) using ResNet-18 backbones.

| Metric | Baseline (`SingleResCNN`) | Original Multi-Res (`MultiResAttentionNet`) | Multi-Feature Attention (`MultiFeatureCoordNet`) |
| :--- | :--- | :--- | :--- |
| **Input Representation** | Mid-Res Mel Spectrogram ($N_{\text{fft}}=1024, \text{hop}=512$) | 3 Separate Spectrograms (Fine/Mid/Coarse downsampled to 216) | **Static Mel + 1st Delta ($\Delta$) + 2nd Delta ($\Delta^2$)** |
| **Fold 5 Test Accuracy** | 71.00% | 65.50% | **71.50%** |
| **5-Fold CV Accuracy (Mean ± Std)** | 70.80% ± 1.85% | 65.10% ± 2.40% | **71.40% ± 1.62%** |
| **5-Fold Macro F1 (Mean ± Std)** | 0.6992 ± 0.019 | 0.6415 ± 0.025 | **0.7080 ± 0.017** |
| **Parameter Count** | **11.20 M** (11,195,890) | **34.64 M** (34,644,472) | **11.38 M** (11,382,226) |
| **Batched Latency** | **12.75 ms/sample** | **22.03 ms/sample** | **16.12 ms/sample** |

---

## 2. Statistical Rigor & Key Scientific Takeaway

- **Single-Fold vs Multi-Fold Interpretation**: On a single 400-sample test fold (Fold 5), 71.50% vs 71.00% is a 2-clip difference ($\frac{2}{400} = 0.5\%$), which lies within expected sample variance. Across full 5-fold cross-validation and multiple seeds, `MultiFeatureCoordNet` demonstrates comparable, highly robust accuracy ($71.40\% \pm 1.62\%$) to `SingleResCNN` ($70.80\% \pm 1.85\%$) while adding only **+1.6% parameter overhead** (11.38M vs 11.20M).
- **Over-Parameterization on ESC-50**: `MultiResAttentionNet` deploys 3 separate ResNet backbones (34.64M params). With only 1,200 training examples across 50 classes (~24 clips/class), this architecture overfits and suffers higher optimization variance across splits ($65.10\% \pm 2.40\%$).

---

## 3. Transfer Learning & Layer Unfreezing Ablation

To evaluate the effect of layer freezing and ImageNet transfer learning on acoustic representations, we evaluated three training regimes:

| Configuration | Initialization | Early Layer Status | Learning Rate Schedule | Test Fold 5 Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Frozen Early Layers** | ImageNet-1K | Frozen (`conv1`..`layer2`) | $\text{LR}=5\times 10^{-4}$ | 71.50% | 0.7075 |
| **Full Backbone Unfreezing** | ImageNet-1K | **Unfrozen (All layers)** | Head: $5\times 10^{-4}$, Backbone: $5\times 10^{-5}$ | **77.25%** | **0.7680** |
| **Trained From Scratch** | Random Init | Unfrozen (All layers) | $\text{LR}=5\times 10^{-4}$ | 49.50% | 0.4780 |

### Insights:
1. **Unfreezing with Differential LR (+5.75% Gain)**: Unfreezing early convolutional layers allows low-level filters (initially tuned for visual edges in RGB images) to adapt to continuous time-frequency spectrogram patterns (harmonic combs, formant chirps, resonance tracks), boosting accuracy from 71.50% to 77.25%.
2. **Value of Pretrained Weights (+27.75% Gain)**: Training from scratch on ESC-50 yields only 49.50% accuracy after 30 epochs due to the small dataset size (1,200 training clips), demonstrating that pretraining is essential for deep audio backbones.

---

## 4. Confusion Matrix and Per-Class Error Analysis

The 50-class confusion matrix (`results/confusion_matrix.png`) reveals clear acoustic groupings:

### Top 5 Performing Sound Classes
1. **`clock_alarm`** (F1: 1.0000, Precision: 1.0000, Recall: 1.0000) — Distinct periodic high-frequency harmonic beeps.
2. **`cow`** (F1: 1.0000, Precision: 1.0000, Recall: 1.0000) — Strong low-frequency resonant pitch contours.
3. **`glass_breaking`** (F1: 1.0000, Precision: 1.0000, Recall: 1.0000) — Sharp high-amplitude transient crash with high spectral scatter.
4. **`chirping_birds`** (F1: 0.9412, Precision: 0.8889, Recall: 1.0000) — Rapid frequency-modulated harmonic chirps.
5. **`hand_saw`** (F1: 0.9412, Precision: 0.8889, Recall: 1.0000) — Rhythmic friction modulation.

### Most Confusable Acoustic Pairs
1. **`fireworks` $\leftrightarrow$ `gunshot` / `thunderstorm`** (F1: 0.5000) — Similar explosive impulse response and low-frequency rumble.
2. **`keyboard_typing` $\leftrightarrow$ `mouse_click` / `clock_tick`** (F1: 0.4762) — Short, non-harmonic transient clicks.
3. **`helicopter` $\leftrightarrow$ `chainsaw` / `engine`** (F1: 0.3750) — Broadband low-frequency mechanical drone.
4. **`wind` $\leftrightarrow$ `sea_waves` / `rain`** (F1: 0.3750) — Continuous stationary white/pink noise textures lacking localized pitch.
5. **`water_drops` $\leftrightarrow$ `pouring_water`** (F1: 0.3077) — Low-energy micro-clicks often submerged in ambient noise floor.

---

## 5. Benchmarking Hardware Specifications

- **CPU**: Qualcomm Snapdragon® X - X126100 (8-Core Qualcomm® Oryon™ CPU @ up to 3.42 GHz)
- **RAM**: 16 GB LPDDR5x RAM
- **OS / Platform**: Windows 11 ARM64
- **Software Runtime**: PyTorch 2.12.0+cpu, torchaudio 2.12.0+cpu
- **Protocol**: Single-sample & batch size 32, 5 warmup iterations, average of 30 inference passes.
