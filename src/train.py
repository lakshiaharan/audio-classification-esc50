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
        args.data_root, batch_size=args.batch_size, test_fold=test_fold, val_fold=val_fold)
    n_classes = len(classes)

    if args.model == 'multires':
        model = MultiResAttentionNet(n_classes, pretrained=not args.no_pretrained).to(device)
    elif args.model == 'multifeature':
        model = MultiFeatureCoordNet(n_classes, pretrained=not args.no_pretrained).to(device)
    else:
        model = SingleResCNN(n_classes, pretrained=not args.no_pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
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

        print(f"[{args.model} fold(test={test_fold})] epoch {epoch:02d}/{args.epochs:02d} "
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
    }
    return result, best_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--model', choices=['multires', 'baseline', 'multifeature'], default='multifeature')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--mixup', action='store_true')
    p.add_argument('--no_pretrained', action='store_true')
    p.add_argument('--cv', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)

    if args.cv:
        fold_results = []
        for test_fold in [1, 2, 3, 4, 5]:
            val_fold = test_fold - 1 if test_fold > 1 else 5
            metrics, best_state = train_one_split(args, device, test_fold, val_fold)
            fold_results.append(metrics)
            torch.save(best_state, f'results/{args.model}_fold{test_fold}_best.pt')

        import statistics
        accs = [r['test_accuracy'] for r in fold_results]
        f1s = [r['test_macro_f1'] for r in fold_results]

        summary = {
            'model': args.model,
            'per_fold': fold_results,
            'mean_test_accuracy': statistics.mean(accs),
            'std_test_accuracy': statistics.stdev(accs),
            'mean_test_macro_f1': statistics.mean(f1s),
            'std_test_macro_f1': statistics.stdev(f1s),
            'num_params': fold_results[0]['num_params'],
            'latency_ms_per_sample': fold_results[0]['latency_ms_per_sample'],
            'epochs': args.epochs,
        }
        with open(f'results/{args.model}_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n5-fold CV test accuracy: {summary['mean_test_accuracy']:.4f} +/- {summary['std_test_accuracy']:.4f}")
        print(f"5-fold CV macro-F1: {summary['mean_test_macro_f1']:.4f} +/- {summary['std_test_macro_f1']:.4f}")

    else:
        metrics, best_state = train_one_split(args, device, test_fold=5, val_fold=4)
        metrics['model'] = args.model
        metrics['epochs'] = args.epochs
        with open(f'results/{args.model}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        torch.save(best_state, f'results/{args.model}_best.pt')

        print(f"\nFinal test accuracy: {metrics['test_accuracy']:.4f} | macro-F1: {metrics['test_macro_f1']:.4f}")
        print(f"Params: {metrics['num_params']:,} | Latency: {metrics['latency_ms_per_sample']:.2f} ms/sample")


if __name__ == '__main__':
    main()
