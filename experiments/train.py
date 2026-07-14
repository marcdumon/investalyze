import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(x: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def macro_accuracy(preds: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        total[c] = mask.sum()
        correct[c] = (preds[mask] == c).sum()
    has_support = total > 0
    return (correct[has_support] / total[has_support]).mean().item()


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def new_best_tracker() -> dict[str, dict]:
    return {
        'train_loss': {'epoch': -1, 'value': float('inf'), 'state': None},
        'valid_loss': {'epoch': -1, 'value': float('inf'), 'state': None},
        'train_acc': {'epoch': -1, 'value': -float('inf'), 'state': None},
        'valid_acc': {'epoch': -1, 'value': -float('inf'), 'state': None},
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    class_weights: np.ndarray,
    n_classes: int,
    device: str,
    history: dict[str, list[float]] | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    scheduler_metric: str = 'valid',
    best: dict[str, dict] | None = None,
    early_stop_patience: int | None = None,
) -> tuple[dict[str, list[float]], dict[str, dict]]:
    model.to(device)
    weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    if history is None:
        history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}
    if best is None:
        best = new_best_tracker()
    start_epoch = len(history['train_loss'])
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_preds.append(logits.argmax(1).detach().cpu())
            train_labels.append(yb.cpu())
        train_loss /= len(train_loader.dataset)
        train_acc = macro_accuracy(torch.cat(train_preds), torch.cat(train_labels), n_classes)

        model.eval()
        valid_loss, valid_preds, valid_labels = 0.0, [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                valid_loss += loss.item() * len(yb)
                valid_preds.append(logits.argmax(1).cpu())
                valid_labels.append(yb.cpu())
        valid_loss /= len(valid_loader.dataset)
        valid_acc = macro_accuracy(torch.cat(valid_preds), torch.cat(valid_labels), n_classes)

        if scheduler is not None:
            scheduler.step(train_loss if scheduler_metric == 'train' else valid_loss)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        if train_loss < best['train_loss']['value']:
            best['train_loss'] = {'epoch': epoch, 'value': train_loss, 'state': _clone_state(model)}
        if valid_loss < best['valid_loss']['value']:
            best['valid_loss'] = {'epoch': epoch, 'value': valid_loss, 'state': _clone_state(model)}
        if train_acc > best['train_acc']['value']:
            best['train_acc'] = {'epoch': epoch, 'value': train_acc, 'state': _clone_state(model)}
        if valid_acc > best['valid_acc']['value']:
            best['valid_acc'] = {'epoch': epoch, 'value': valid_acc, 'state': _clone_state(model)}

        lr = optimizer.param_groups[0]['lr']
        best_epochs = (
            f"(t:{best['train_loss']['epoch'] + 1}/{best['train_acc']['epoch'] + 1} - "
            f"v:{best['valid_loss']['epoch'] + 1}/{best['valid_acc']['epoch'] + 1})"
        )
        print(f'epoch {epoch + 1}/{start_epoch + epochs}  train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} '
              f'train_acc={train_acc:.4f} valid_acc={valid_acc:.4f} lr={lr:.2e} {best_epochs}')

        if early_stop_patience is not None and epoch - best['valid_loss']['epoch'] >= early_stop_patience:
            print(f"early stopping: no valid_loss improvement since epoch {best['valid_loss']['epoch'] + 1}")
            break

    return history, best


def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels)
