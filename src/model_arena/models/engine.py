"""Shared numerical execution for forecasters with different data adapters."""

import numpy as np
import torch

from ..scoring import pinball_error


def pinball_tensor(error, quantiles):
    probabilities = error.new_tensor(quantiles)
    return pinball_error(error, probabilities, maximum=torch.maximum)


def train_quantiles(network, optimizer, schedule, *, epochs, batch_size, quantiles, batch, on_epoch,
                    accumulate_double=True, point=False, point_loss=None):
    """Fit a network using the profile's schedule and normalized input/target adapter."""
    for epoch in range(1, epochs + 1):
        network.train()
        refs = schedule()
        total = None
        overall = 0.0
        for start in range(0, len(refs), batch_size):
            batch_refs = refs[start:start + batch_size]
            inputs, truth = batch(batch_refs)
            optimizer.zero_grad()
            if point_loss is not None:
                elements = point_loss(*inputs, truth.unsqueeze(-1))
            else:
                error = truth.unsqueeze(-1) - network(*inputs)
                elements = error.abs() if point else pinball_tensor(error, quantiles)
            loss = elements.mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            reduced = elements.detach()
            if accumulate_double:
                reduced = reduced.double()
            counts = reduced.mean(dim=(0, 1, 3)).cpu().numpy() * len(batch_refs)
            if total is None:
                total = np.zeros(len(counts))
            total += counts
            overall += loss.detach().item() * len(batch_refs)
        if total is None:
            raise ValueError("No training windows")
        on_epoch(epoch, overall / len(refs), total / len(refs))


def predict_batches(network, origins, *, batch_size, batch):
    """Predict in batches and return native values plus optional references."""
    previous = network.training
    values, auxiliary = [], []
    network.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(origins), batch_size):
                inputs, scales, extra = batch(origins[start:start + batch_size])
                predictions = network(*inputs).cpu().numpy().astype(np.float64)
                values.append(predictions * scales)
                if extra is not None:
                    auxiliary.append(extra)
    finally:
        network.train(previous)
    predictions = np.concatenate(values)
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite predictions")
    return predictions, np.concatenate(auxiliary) if auxiliary else None


def temporal_folds(size, *, window, horizon, count, step):
    first = size - horizon - (count - 1) * step
    if first < window + horizon:
        raise ValueError("Insufficient history for temporal protocol")
    return [(f"F{i + 1}", first + i * step) for i in range(count)]
