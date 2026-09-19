"""
src/evaluate.py - Comprehensive Evaluation Suite for ESC-50 Audio Classification
Computes:
  1. Overall Accuracy, Macro F1, Weighted F1, Top-1/3/5 Accuracy
  2. 50-Class Confusion Matrix Heatmap (results/confusion_matrix.png)
  3. Per-Class Precision, Recall, and F1-Scores (results/per_class_f1.json)
  4. Top-5 and Bottom-5 acoustic categories analysis
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data import get_dataloaders, SAMPLE_RATE
from models import MultiResAttentionNet, SingleResCNN, MultiFeatureCoordNet


def evaluate_model(
    model_type="multifeature",
    checkpoint_path=None,
    data_root="ESC-50",
    test_fold=5,
    val_fold=4,
    device=None,
    output_dir="results",
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    _, _, test_loader, classes = get_dataloaders(
        data_root, batch_size=32, test_fold=test_fold, val_fold=val_fold, feature_type=model_type
    )
    n_classes = len(classes)

    ckpt_path = checkpoint_path or f"{output_dir}/{model_type}_best.pt"
    if not os.path.exists(ckpt_path):
        # Fallback to standard name
        alt_ckpt = f"{output_dir}/{model_type}_best.pt"
        if os.path.exists(alt_ckpt):
            ckpt_path = alt_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"[Evaluation] Loading model '{model_type}' from checkpoint: {ckpt_path}")

    if model_type == "multires":
        model = MultiResAttentionNet(n_classes=n_classes, pretrained=False, freeze_early=False).to(device)
    elif model_type == "multifeature":
        model = MultiFeatureCoordNet(n_classes=n_classes, pretrained=False, freeze_early=False).to(device)
    else:
        model = SingleResCNN(n_classes=n_classes, pretrained=False, freeze_early=False).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_targets = []
    all_top3 = []
    all_top5 = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)

            top1 = probs.argmax(dim=1).cpu().numpy()
            top3 = torch.topk(probs, k=min(3, n_classes), dim=1).indices.cpu().numpy()
            top5 = torch.topk(probs, k=min(5, n_classes), dim=1).indices.cpu().numpy()

            all_preds.extend(top1)
            all_targets.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

            for target, t3, t5 in zip(y.numpy(), top3, top5):
                all_top3.append(target in t3)
                all_top5.append(target in t5)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Overall Metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    top3_acc = np.mean(all_top3)
    top5_acc = np.mean(all_top5)

    # 2. Detailed Classification Report & Per-Class F1
    report_dict = classification_report(
        all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0
    )

    per_class_metrics = {}
    for idx, name in enumerate(classes):
        m = report_dict.get(name, {})
        per_class_metrics[name] = {
            "class_id": idx,
            "precision": round(m.get("precision", 0.0), 4),
            "recall": round(m.get("recall", 0.0), 4),
            "f1_score": round(m.get("f1-score", 0.0), 4),
            "support": int(m.get("support", 0)),
        }

    # Save per-class F1 json
    f1_json_path = os.path.join(output_dir, "per_class_f1.json")
    with open(f1_json_path, "w") as f:
        json.dump(
            {
                "model": model_type,
                "checkpoint": ckpt_path,
                "test_fold": test_fold,
                "test_accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "top3_accuracy": round(top3_acc, 4),
                "top5_accuracy": round(top5_acc, 4),
                "per_class": per_class_metrics,
            },
            f,
            indent=2,
        )
    print(f"[Evaluation] Saved per-class metrics to {f1_json_path}")

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(n_classes)))
    plt.figure(figsize=(18, 15))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Sample Count"},
        linewidths=0.5,
        linecolor="#e2e8f0",
    )
    plt.title(
        f"ESC-50 50-Class Confusion Matrix — {model_type.upper()} (Test Fold {test_fold})\n"
        f"Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1:.4f} | Top-5 Accuracy: {top5_acc*100:.2f}%",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrix heatmap to {cm_path}")

    # 4. Print Summary Table
    print("\n" + "=" * 78)
    print(f"{'ESC-50 EVALUATION RESULTS SUMMARY':^78}")
    print("=" * 78)
    print(f"Model Architecture  : {model_type}")
    print(f"Evaluation Split    : Fold {test_fold} (N = {len(all_targets)} samples)")
    print(f"Top-1 Accuracy      : {acc*100:.2f}%")
    print(f"Top-3 Accuracy      : {top3_acc*100:.2f}%")
    print(f"Top-5 Accuracy      : {top5_acc*100:.2f}%")
    print(f"Macro F1 Score      : {macro_f1:.4f}")
    print(f"Weighted F1 Score   : {weighted_f1:.4f}")
    print("-" * 78)

    # Sort classes by F1
    sorted_by_f1 = sorted(per_class_metrics.items(), key=lambda item: item[1]["f1_score"], reverse=True)
    print("\nTop 5 Performing Classes (Highest F1):")
    for name, m in sorted_by_f1[:5]:
        print(f"  • {name:<20} F1: {m['f1_score']:.4f} (Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f})")

    print("\nBottom 5 Challenging Classes (Acoustically Confusable):")
    for name, m in sorted_by_f1[-5:]:
        print(f"  • {name:<20} F1: {m['f1_score']:.4f} (Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f})")
    print("=" * 78 + "\n")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio classification checkpoints on ESC-50")
    parser.add_argument("--model", choices=["multifeature", "baseline", "multires"], default="multifeature")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_root", default="ESC-50")
    parser.add_argument("--test_fold", type=int, default=5)
    parser.add_argument("--val_fold", type=int, default=4)
    args = parser.parse_args()

    evaluate_model(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
    )
