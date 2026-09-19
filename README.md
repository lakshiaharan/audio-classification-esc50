# Environmental Sound Classification on ESC-50

[![CI](https://github.com/lakshiaharan/audio-classification-esc50/actions/workflows/ci.yml/badge.svg)](https://github.com/lakshiaharan/audio-classification-esc50/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776ab.svg)](https://www.python.org/)
[![Vercel Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)](https://audio-classification-esc50.vercel.app)

> **GitHub Repository About Settings**:
> - **Description**: *An end-to-end deep learning framework for environmental sound classification comparing multi-resolution STFTs, differential spectral velocities (Δ, Δ²), and coordinate attention on ESC-50.*
> - **Topics**: `audio-classification`, `esc-50`, `pytorch`, `deep-learning`, `spectrogram-analysis`, `coordinate-attention`, `sound-recognition`

---

<div align="center">
  <img src="docs/demo.gif" alt="SoundPulse Studio Demo" width="800px" style="border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);" />
  <p><em>Interactive SoundPulse Studio interface with real-time waveform rendering, preset auditioning, and classification.</em></p>
</div>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Inference Engine Architecture](#inference-engine-architecture)
- [Experimental Results & 5-Fold Cross-Validation](#experimental-results--5-fold-cross-validation)
- [Validation Protocol & Leak-Free Split Methodology](#validation-protocol--leak-free-split-methodology)
- [Ablation Study: Transfer Learning & Layer Unfreezing](#ablation-study-transfer-learning--layer-unfreezing)
- [Detailed Results: Confusion Matrix & Per-Class F1](#detailed-results-confusion-matrix--per-class-f1)
- [Hardware & Latency Benchmark](#hardware--latency-benchmark)
- [Project Structure](#project-structure)
- [Installation & Dependency Guide](#installation--dependency-guide)
- [Quickstart: Training, Evaluation & CLI](#quickstart-training-evaluation--cli)
- [Pretrained Checkpoints & Releases](#pretrained-checkpoints--releases)
- [Testing & CI Pipeline](#testing--ci-pipeline)
- [License & Citation](#license--citation)

---

## Project Overview

Environmental sound classification (ESC) presents distinctive challenges compared to speech recognition or music information retrieval:
- **Acoustic Diversity**: Broad spectral distributions spanning natural ambient textures (rain, sea waves), transient percussive onsets (door knocks, gunshots), and harmonic acoustic signatures (dog barks, sirens).
- **Temporal Event Dynamics**: Salient acoustic cues often occur in brief bursts within longer audio clips.
- **Limited Dataset Scale**: Standard ESC-50 contains 2,000 clips (40 clips per category across 50 classes), making over-parameterized models prone to overfitting.

### Architectures Implemented & Evaluated
1. **Multi-Feature Differential Network (`MultiFeatureCoordNet`) [Headline Model]**: 3-channel input combining **Static Log-Mel + 1st Temporal Derivative ($\Delta$, Velocity) + 2nd Temporal Derivative ($\Delta^2$, Acceleration)** with decoupled 1D coordinate attention.
2. **Single-Resolution Baseline (`SingleResCNN`)**: Standard ResNet-18 backbone processing single 64-band Log-Mel Spectrograms ($N_{\text{fft}}=1024, \text{hop}=512$).
3. **Multi-Resolution Attention Network (`MultiResAttentionNet`)**: A 3-branch network processing Fine ($\text{hop}=128$), Mid ($\text{hop}=512$), and Coarse ($\text{hop}=1024$) resolutions simultaneously with spatial time-frequency attention and dynamic gating.

---

## Inference Engine Architecture

To balance deep learning model complexity with high-availability web deployment, this project implements a transparent multi-tier execution model:

| Component | Runtime Environment | Model Engine | Primary Goal |
| :--- | :--- | :--- | :--- |
| **Local PyTorch Server (`server.py`)** | Local Python environment (CPU / CUDA) | Full PyTorch Deep Learning Checkpoints (`MultiFeatureCoordNet`, `SingleResCNN`) | Full-fidelity neural network inference with GPU acceleration support (**77.3% test accuracy**) |
| **CLI Predictor (`src/predict.py`)** | Local CLI (Torch / TorchAudio) | Native PyTorch `.pt` Checkpoint Weights | Standalone audio batch scoring & research verification |
| **Web Studio Demo (Vercel)** | Edge / Browser (Web Audio API) | Zero-Binary Spectral Centroid Matching (`api/mel_signatures.py`) | Sub-20ms edge inference without heavy 1GB+ C++ binary wheels on serverless |

> [!NOTE]
> **Web Demo vs. Local PyTorch Engine**:
> Vercel Serverless free-tier deployments enforce a 250 MB uncompressed / 50 MB compressed bundle limit and lack pre-compiled C++ CUDA/PyTorch runtimes. The online demo at [audio-classification-esc50.vercel.app](https://audio-classification-esc50.vercel.app) uses client-side WebAudio feature extraction with precomputed 32-band spectral centroid matching (~7.3% Top-1 / ~25% Top-5 accuracy on Fold 5). To run the full **77.3% PyTorch deep neural networks** interactively, launch the local server with `python server.py`.

---

## Experimental Results & 5-Fold Cross-Validation

### Headline Architecture Benchmark (Fine-Tuned Unfrozen Backbones)

All models were evaluated across 5-fold cross-validation with 3 random seeds (seeds 42, 123, 999; $N = 15$ runs total per architecture). Each run trained for 30 epochs with AdamW optimizer, Cosine Annealing learning rate schedule, differential learning rates (Head $\text{LR}=5\times 10^{-4}$, Backbone $\text{LR}=5\times 10^{-5}$), SpecAugment, and Mixup ($\alpha = 0.3$).

$$\pm \text{ denotes the sample standard deviation across all 15 fold-seed iterations.}$$

| Model Architecture | Input Features | Parameters | Fold 5 Accuracy | 5-Fold CV Accuracy (Mean ± Std) | 5-Fold Macro F1 (Mean ± Std) | Latency (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-Feature CoordNet (Headline)** | **Static Mel + $\Delta$ + $\Delta^2$** | **11.38 M** | **77.25%** | **76.80% ± 1.45%** | **0.7635 ± 0.016** | **16.12 ms** |
| **Baseline SingleResCNN** | Static Log-Mel Spectrogram | 11.20 M | 76.50% | 76.10% ± 1.55% | 0.7550 ± 0.018 | 12.75 ms |
| **Multi-Resolution AttentionNet** | 3-Branch Multi-Scale STFT | 34.64 M | 68.75% | 68.20% ± 2.10% | 0.6740 ± 0.022 | 22.03 ms |

![Benchmark Comparison](results/comparison.png)

### Key Scientific Takeaways
- **Comparable Accuracy at +1.6% Parameters**: Across 15 fold-seed evaluation runs, `MultiFeatureCoordNet` demonstrates comparable performance ($76.80\% \pm 1.45\%$) to `SingleResCNN` ($76.10\% \pm 1.55\%$) with only +1.6% parameter overhead (11.38M vs 11.20M parameters).
- **Multi-Branch Resolution Degradation**: `MultiResAttentionNet` (34.64M params) underperformed ($68.20\% \pm 2.10\%$). This is likely due to spatial downsampling aliasing (forcing 1,723 fine-hop time frames into a fixed 216-frame grid via bilinear downsampling applied an 8× compression), combined with over-parameterization on the small training set (~24 clips per class).
- **Differential Features Preserve Temporal Onsets**: Providing explicit $\Delta$ (velocity) and $\Delta^2$ (acceleration) channels injects first- and second-order time derivatives directly into a single 3-channel backbone without requiring multi-branch backbones or spatial downsampling.

---

## Validation Protocol & Leak-Free Split Methodology

To ensure strict zero test leakage and prevent optimistic performance estimates:
1. **3-Way Split Protocol**: In each cross-validation fold $k \in \{1, 2, 3, 4, 5\}$:
   - **Training Set (3 Folds, 1,200 clips)**: Used exclusively for gradient optimization.
   - **Validation Set (1 Fold, 400 clips)**: Used solely for monitoring validation loss/accuracy and model checkpointing (`*_best.pt` saves state at peak validation accuracy).
   - **Test Set (1 Fold, 400 clips)**: Held out completely untouched throughout training and evaluated strictly once at the end using the checkpoint selected on the validation set.
2. **Zero Test Contamination**: Test fold data is never seen during gradient updates, learning rate scheduling, or checkpoint selection.

---

## Ablation Study: Transfer Learning & Layer Unfreezing

To quantify the impact of visual transfer learning and convolutional layer freezing, we evaluated three training regimes across 5-fold cross-validation (3 random seeds, $N=15$ runs each):

| Configuration | Initialization | Early Layer Status | Learning Rate Schedule | 5-Fold CV Accuracy (Mean ± Std) | Fold 5 Accuracy | Macro F1 (Fold 5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Backbone Unfreezing (Headline)** | ImageNet-1K | **Unfrozen (All layers)** | Head: $5\times 10^{-4}$, Backbone: $5\times 10^{-5}$ | **76.80% ± 1.45%** | **77.25%** | **0.7680** |
| **Frozen Early Layers** | ImageNet-1K | Frozen (`conv1`..`layer2`) | $\text{LR}=5\times 10^{-4}$ | 71.40% ± 1.62% | 71.50% | 0.7075 |
| **Trained From Scratch (Ablation)** | Random Init | Unfrozen (All layers) | $\text{LR}=5\times 10^{-4}$ | 48.60% ± 2.15% | 49.50% | 0.4780 |

### Analysis
1. **Domain Adaptation (+5.40% CV Gain)**: Unfreezing early convolutional layers with a conservative differential learning rate ($0.1\times$ head LR) enables low-level visual edge filters to adapt to continuous time-frequency spectrogram patterns (harmonic combs, formant chirps, resonance tracks).
2. **Transfer Learning Value**: Training from scratch yields $48.60\% \pm 2.15\%$ accuracy. Note that a fixed 30-epoch training budget is likely sub-optimal for scratch training; thus, the ~28.2-point gap represents an upper bound on the ImageNet transfer learning advantage on low-resource environmental sound datasets.

---

## Detailed Results: Confusion Matrix & Per-Class F1

The complete 50-class confusion matrix and per-class performance metrics are computed using `src/evaluate.py`:

<div align="center">
  <img src="results/confusion_matrix.png" alt="ESC-50 Confusion Matrix" width="750px" style="border-radius: 6px; box-shadow: 0 4px 16px rgba(0,0,0,0.1);" />
  <p><em>50-class confusion matrix for MultiFeatureCoordNet on ESC-50 Test Fold 5 (N = 400 test samples).</em></p>
</div>

> [!NOTE]
> **Sample-Size Noise Caveat on Per-Class Metrics**:
> In ESC-50, each test fold contains exactly **8 audio clips per category** ($N=8$). Consequently, a single misclassified sample shifts class recall by **12.5%**, so per-class numbers should be interpreted with awareness of single-sample sensitivity.

### Per-Class F1 Score Breakdown

| Class Category | Sound Class | Precision | Recall | F1 Score | Verified Acoustic Misclassification Patterns |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Top Performing** | `clock_alarm` | 1.0000 | 1.0000 | **1.0000** | Distinct periodic high-frequency harmonic beeps (0 errors) |
| | `cow` | 1.0000 | 1.0000 | **1.0000** | Low-frequency resonant pitch contours (0 errors) |
| | `glass_breaking` | 1.0000 | 1.0000 | **1.0000** | Sharp high-amplitude transient crash + high spectral scatter (0 errors) |
| | `chirping_birds` | 0.8889 | 1.0000 | **0.9412** | High-pitch FM chirps with clear harmonic separation |
| | `hand_saw` | 0.8889 | 1.0000 | **0.9412** | Distinct periodic friction modulation |
| **Challenging / Confusable** | `fireworks` | 0.7500 | 0.3750 | **0.5000** | Confused with `clapping` (3 clips) due to impulsive burst similarity |
| | `keyboard_typing` | 0.3846 | 0.6250 | **0.4762** | Confused with `mouse_click` and `clock_tick` due to micro-click transients |
| | `helicopter` | 0.3750 | 0.3750 | **0.3750** | Confused with `airplane` (2 clips) due to low-frequency engine rumble |
| | `wind` | 0.3750 | 0.3750 | **0.3750** | Confused with `siren` (2 clips) and `vacuum_cleaner` (2 clips) due to broadband noise |
| | `water_drops` | 0.4000 | 0.2500 | **0.3077** | Confused with `clock_tick` (2 clips) due to periodic quiet transients |

*Complete metrics for all 50 categories are serialized in [`results/per_class_f1.json`](results/per_class_f1.json).*

---

## Hardware & Latency Benchmark

All latency and inference benchmarks were measured with batch warmup iterations under identical system conditions:

- **Processor**: Qualcomm Snapdragon® X - X126100 (8-Core Qualcomm® Oryon™ CPU @ up to 3.42 GHz)
- **Host Architecture**: ARM64 (aarch64), Windows 11
- **System Memory**: 16 GB LPDDR5x RAM
- **Software Runtime**: PyTorch 2.12.0+cpu, torchaudio 2.12.0+cpu
- **Benchmark Protocol**: 5 warmup iterations, 30 timed passes with batch size $N=32$, normalized to single-sample latency ($\text{ms} / \text{sample}$).

---

## Project Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions automated unit test workflow
├── api/                            # Vercel serverless functions
│   ├── index.py                    # Serverless HTTP request router
│   ├── mel_signatures.py           # Precomputed 32-band spectral signatures
│   └── metrics.py                  # Serverless metrics endpoint
├── docs/                           # Documentation assets & deep-dive reports
│   ├── demo.gif                    # UI walkthrough preview GIF
│   └── results.md                  # In-depth architectural analysis & engineering logs
├── public/                         # SoundPulse Studio frontend
│   ├── index.html                  # Studio user interface & transparency notice
│   ├── style.css                   # Glassmorphic responsive CSS
│   └── app.js                      # WebAudio recording & client inference
├── results/                        # Evaluation artifacts
│   ├── comparison.png              # 4-panel architecture comparison chart
│   ├── confusion_matrix.png        # 50-class confusion matrix heatmap
│   ├── per_class_f1.json           # Complete per-class precision/recall/F1 metrics
│   └── multifeature_metrics.json
├── src/                            # PyTorch deep learning codebase
│   ├── data.py                     # Audio loader, Mel transforms, deltas & dataset caching
│   ├── models.py                   # SingleResCNN, MultiResAttentionNet, MultiFeatureCoordNet
│   ├── train.py                    # Training loop, Mixup, 5-Fold CV & unfreezing differential LR
│   ├── evaluate.py                 # Comprehensive evaluation suite & confusion matrix generator
│   ├── predict.py                  # CLI audio classification utility
│   └── compare_results.py          # Benchmark plotting script
├── tests/                          # Test suite
│   ├── test_models.py              # Model forward pass, shapes, and gradient unit tests
│   ├── test_data.py                # Audio extraction, transforms, and deltas unit tests
│   └── integration_test_server.py  # Server integration test (runs with active local server)
├── LICENSE                         # MIT License
├── requirements.txt                # Minimal Vercel deployment dependencies (<1MB bundle)
├── requirements-local.txt          # Full PyTorch training, evaluation & server dependencies
└── server.py                       # Local Python deep learning server
```

---

## Installation & Dependency Guide

### `requirements-local.txt` vs `requirements.txt`
- **`requirements-local.txt`**: Used for local PyTorch model training, full unfreezing, evaluation, and running `server.py`. Installs PyTorch, TorchAudio, torchvision, Pandas, Scikit-Learn, Matplotlib, and Seaborn.
- **`requirements.txt`**: Kept minimal with zero heavy binary dependencies to satisfy Vercel Serverless size constraints (<50MB package limit, <5s deployment build time).

```bash
# 1. Clone repository
git clone https://github.com/lakshiaharan/audio-classification-esc50.git
cd audio-classification-esc50

# 2. Install full local dependencies
pip install -r requirements-local.txt

# 3. (Optional) Clone ESC-50 dataset for local training
git clone --depth 1 https://github.com/karolpiczak/ESC-50.git
```

---

## Quickstart: Training, Evaluation & CLI

### 1. Train Models Locally

**Train Headline Multi-Feature Model with Full Backbone Unfreezing (Differential LR)**:
```bash
python src/train.py --data_root ESC-50 --model multifeature --epochs 30 --mixup --unfreeze --lr 5e-4 --backbone_lr 5e-5
```

**Run 5-Fold Cross-Validation across 3 Random Seeds**:
```bash
python src/train.py --data_root ESC-50 --model multifeature --epochs 30 --mixup --unfreeze --lr 5e-4 --backbone_lr 5e-5 --cv --seeds 42 123 999
```

**Run Frozen Baseline Training**:
```bash
python src/train.py --data_root ESC-50 --model multifeature --epochs 30 --mixup
```

**Run Training From Scratch Ablation (No ImageNet Pretraining)**:
```bash
python src/train.py --data_root ESC-50 --model multifeature --epochs 30 --mixup --no_pretrained
```

### 2. Comprehensive Evaluation & Confusion Matrix
```bash
python src/evaluate.py --model multifeature --data_root ESC-50 --test_fold 5
```
*Outputs overall accuracy, top-3/top-5 metrics, saves `results/confusion_matrix.png`, and exports `results/per_class_f1.json`.*

### 3. CLI Prediction on Arbitrary `.wav` Audio
```bash
python src/predict.py ESC-50/audio/1-100032-A-0.wav --model multifeature
```

### 4. Launch Local Interactive Web Studio
```bash
python server.py
```
Open [http://localhost:8000](http://localhost:8000) to test 50 preset categories, upload `.wav` audio files, or classify microphone recordings with local PyTorch model execution.

---

## Pretrained Checkpoints & Releases

Trained model checkpoints are available in the official GitHub Release:

| Checkpoint File | Architecture | Training Regime | Test Fold 5 Accuracy | 5-Fold CV (Mean ± Std) | Size | Download |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `multifeature_best.pt` | MultiFeatureCoordNet | Fine-Tuned (Headline) | **77.25%** | **76.80% ± 1.45%** | ~45.6 MB | [GitHub Releases](https://github.com/lakshiaharan/audio-classification-esc50/releases) |
| `baseline_best.pt` | SingleResCNN | Fine-Tuned Baseline | 76.50% | 76.10% ± 1.55% | ~44.8 MB | [GitHub Releases](https://github.com/lakshiaharan/audio-classification-esc50/releases) |
| `multires_best.pt` | MultiResAttentionNet | Fine-Tuned (3-Branch) | 68.75% | 68.20% ± 2.10% | ~138.8 MB | [GitHub Releases](https://github.com/lakshiaharan/audio-classification-esc50/releases) |

To use downloaded weights:
```bash
mkdir -p results
# Place downloaded .pt files in results/
python src/predict.py your_sound.wav --checkpoint results/multifeature_best.pt
```

---

## Testing & CI Pipeline

Unit tests validate architecture shapes, parameter unfreezing logic, gradient backpropagation, and audio feature extraction transforms without server dependencies:

```bash
python -m unittest discover tests
# or using pytest
pytest tests/test_models.py tests/test_data.py -v
```

GitHub Actions automatically executes the unit test matrix on all pushes and pull requests targeting `main` (Python 3.10 and 3.11).

*To run the server integration test (requires active local server on port 8000):*
```bash
python tests/integration_test_server.py
```

---

## License & Citation

This project is licensed under the [MIT License](LICENSE).

```bibtex
@inproceedings{piczak2015esc,
  title={ESC: Dataset for Environmental Sound Classification},
  author={Piczak, Karol J.},
  booktitle={Proceedings of the 23rd ACM International Conference on Multimedia},
  pages={1015--1018},
  year={2015}
}
```
