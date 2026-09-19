import argparse, json, os, time, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from data import get_dataloaders
from models import MultiResAttentionNet, SingleResCNN, MultiFeatureCoordNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup(x, y, n_classes, alpha=0.3):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1-lam) * x[idx]
    y_oh = torch.zeros(x.size(0), n_classes, device=x.device).scatter_(1, y.unsqueeze(1), 1)
    mixed_y = lam * y_oh + (1-lam) * y_oh[idx]
    return mixed_x, mixed_y


def run_epoch(model, loader, criterion, optimizer, device, train=True, n_classes=50, use_mixup=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if train:
                optimizer.zero_grad()
                if use_mixup:
                    mx, my = mixup(x, y, n_classes)
                    out = model(mx)
                    logp = torch.log_softmax(out, dim=1)
                    loss = -(my * logp).sum(dim=1).mean()
                    preds = out.argmax(1).cpu().tolist()
                    targets = my.argmax(1).cpu().tolist()
                else:
                    out = model(x)
                    loss = criterion(out, y)
                    preds = out.argmax(1).cpu().tolist()
                    targets = y.cpu().tolist()
                loss.backward()
                optimizer.step()
            else:
                out = model(x)
                loss = criterion(out, y)
                preds = out.argmax(1).cpu().tolist()
                targets = y.cpu().tolist()

            total_loss += loss.item() * x.size(0)
            all_preds += preds
            all_labels += targets

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader.dataset), acc, f1


def measure_latency(model, sample_input, device, n_runs=30):
    model.eval()
    sample_input = sample_input.to(device)
    bs = sample_input.size(0)

    with torch.no_grad():
        for _ in range(5):
            model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_runs):
            model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()

    ms_per_batch = (t1 - t0) / n_runs * 1000
    return ms_per_batch / bs


def train_one_split(args, device, test_fold, val_fold):
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_root, batch_size=args.batch_size, test_fold=test_fold, val_fold=val_fold, feature_type=args.model)
    n_classes = len(classes)

    pretrained = not args.no_pretrained
    freeze_early = not args.unfreeze

    if args.model == 'multires':
        model = MultiResAttentionNet(n_classes, pretrained=pretrained, freeze_early=freeze_early).to(device)
    elif args.model == 'multifeature':
        model = MultiFeatureCoordNet(n_classes, pretrained=pretrained, freeze_early=freeze_early).to(device)
    else:
        model = SingleResCNN(n_classes, pretrained=pretrained, freeze_early=freeze_early).to(device)

    criterion = nn.CrossEntropyLoss()
    if hasattr(model, 'get_param_groups') and (args.unfreeze or args.backbone_lr is not None):
        backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr * 0.1
        param_groups = model.get_param_groups(head_lr=args.lr, backbone_lr=backbone_lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device,
                                              train=True, n_classes=n_classes, use_mixup=args.mixup)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device,
                                               train=False, n_classes=n_classes)
        scheduler.step()

        print(f"[{args.model} (pretrained={pretrained}, unfreeze={args.unfreeze}) fold(test={test_fold})] epoch {epoch:02d}/{args.epochs:02d} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}", flush=True)

        if val_acc > best_val_acc or best_state is None:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1 = run_epoch(model, test_loader, criterion, optimizer, device,
                                              train=False, n_classes=n_classes)

    n_params = sum(p.numel() for p in model.parameters())
    sample_x, _ = next(iter(test_loader))
    latency_ms = measure_latency(model, sample_x, device)

    result = {
        'test_fold': test_fold,
        'test_accuracy': test_acc,
        'test_macro_f1': test_f1,
        'best_val_accuracy': best_val_acc,
        'num_params': n_params,
        'latency_ms_per_sample': latency_ms,
        'pretrained': pretrained,
        'unfreeze': args.unfreeze,
    }
    return result, best_state


def main():
    p = argparse.ArgumentParser(description="Train and evaluate audio classification models on ESC-50")
    p.add_argument('--data_root', required=True, help="Path to ESC-50 dataset root")
    p.add_argument('--model', choices=['multires', 'baseline', 'multifeature'], default='multifeature')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4, help="Learning rate (for classifier head or overall model)")
    p.add_argument('--backbone_lr', type=float, default=None, help="Differential learning rate for backbone when unfreezing")
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--mixup', action='store_true', help="Enable Mixup data augmentation")
    p.add_argument('--unfreeze', action='store_true', help="Unfreeze all early backbone layers for full end-to-end fine-tuning")
    p.add_argument('--no_pretrained', action='store_true', help="Train from scratch with random initialization (ablation study)")
    p.add_argument('--cv', action='store_true', help="Run 5-fold cross-validation")
    p.add_argument('--seeds', type=int, nargs='+', default=[42], help="Random seeds to evaluate across (e.g. --seeds 42 123 999)")
    p.add_argument('--seed', type=int, default=None, help="Single random seed (shorthand)")
    args = p.parse_args()

    if args.seed is not None:
        args.seeds = [args.seed]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)

    suffix = ""
    if args.unfreeze:
        suffix += "_unfrozen"
    if args.no_pretrained:
        suffix += "_scratch"

    if args.cv:
        import statistics
        all_seed_results = []
        for seed in args.seeds:
            set_seed(seed)
            print(f"\n================ Running 5-Fold CV with Seed {seed} ================")
            seed_fold_results = []
            for test_fold in [1, 2, 3, 4, 5]:
                val_fold = test_fold - 1 if test_fold > 1 else 5
                metrics, best_state = train_one_split(args, device, test_fold, val_fold)
                metrics['seed'] = seed
                seed_fold_results.append(metrics)
                torch.save(best_state, f'results/{args.model}{suffix}_seed{seed}_fold{test_fold}_best.pt')
            all_seed_results.extend(seed_fold_results)

        all_accs = [r['test_accuracy'] for r in all_seed_results]
        all_f1s = [r['test_macro_f1'] for r in all_seed_results]

        summary = {
            'model': args.model,
            'pretrained': not args.no_pretrained,
            'unfrozen': args.unfreeze,
            'seeds': args.seeds,
            'n_runs': len(all_seed_results),
            'per_run_results': all_seed_results,
            'mean_test_accuracy': statistics.mean(all_accs),
            'std_test_accuracy': statistics.stdev(all_accs) if len(all_accs) > 1 else 0.0,
            'mean_test_macro_f1': statistics.mean(all_f1s),
            'std_test_macro_f1': statistics.stdev(all_f1s) if len(all_f1s) > 1 else 0.0,
            'num_params': all_seed_results[0]['num_params'],
            'latency_ms_per_sample': all_seed_results[0]['latency_ms_per_sample'],
            'epochs': args.epochs,
        }
        with open(f'results/{args.model}{suffix}_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"5-Fold CV Summary across {len(args.seeds)} seed(s) [{args.model}{suffix}]:")
        print(f"Mean Test Accuracy : {summary['mean_test_accuracy']*100:.2f}% ± {summary['std_test_accuracy']*100:.2f}%")
        print(f"Mean Macro-F1      : {summary['mean_test_macro_f1']:.4f} ± {summary['std_test_macro_f1']:.4f}")
        print(f"{'='*70}\n")

    else:
        seed = args.seeds[0]
        set_seed(seed)
        metrics, best_state = train_one_split(args, device, test_fold=5, val_fold=4)
        metrics['model'] = args.model
        metrics['epochs'] = args.epochs
        metrics['seed'] = seed
        with open(f'results/{args.model}{suffix}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        torch.save(best_state, f'results/{args.model}{suffix}_best.pt')

        print(f"\nFinal test accuracy: {metrics['test_accuracy']:.4f} | macro-F1: {metrics['test_macro_f1']:.4f}")
        print(f"Params: {metrics['num_params']:,} | Latency: {metrics['latency_ms_per_sample']:.2f} ms/sample")


if __name__ == '__main__':
    main()
