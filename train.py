import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    size = len(train_loader.dataset)
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        grades = batch['grades'].to(device).float().unsqueeze(1)  # unsqueeze to ensure this is dim [batch_size,1]
        seqs = batch['seqs'].to(device)
        pad_mask = batch['pad_mask'].to(device)

        optimizer.zero_grad()

        # Replace -1 with the model's padding index before passing to model
        seqs[seqs == -1] = model.pad_idx

        inp = seqs[:, :-1]  # the shifted input, excluding the last token
        inp_pad_mask = pad_mask[:, :-1]
        tgt = seqs[:, 1:]  # the target output, excluding BOS token

        out = model(inp, grades=grades, pad_mask=inp_pad_mask)

        # compute prediction error
        # out is (B, T, vocab_size), we need (B*T, vocab_size)
        # seqs is (B, T) we need (B*T)
        out_flat = out.view(-1, out.size(-1))
        tgt_flat = tgt.reshape(-1)
        loss = criterion(out_flat, tgt_flat)

        # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seqs.size(0)

        _, predicted = torch.max(out_flat, 1)
        non_pad_mask = (tgt_flat != model.pad_idx)

        correct += ((predicted == tgt_flat) & non_pad_mask).sum().item()
        total += non_pad_mask.sum().item()

        if batch_idx % (len(train_loader) // 5) == 0:
            print(f"loss: {loss.item():>7f}  [{batch_idx + 1}/{len(train_loader)}]")

    if scheduler is not None:
        try:
            scheduler.step()
        except ZeroDivisionError:
            print("T_max is zero")

    avg_loss = total_loss / size
    accuracy = 100 * (correct / total) if total > 0 else 0

    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            seqs = batch['seqs'].to(device)
            grades = batch['grades'].to(device)
            pad_mask = batch['pad_mask'].to(device)

            out = model(seqs, pad_mask=pad_mask)

            out_flat = out.view(-1, out.size(-1))
            seqs_flat = seqs.view(-1)
            loss = criterion(out_flat, seqs_flat)

            total_loss = total_loss + loss.item() * seqs.size(0)

            _, predicted = torch.max(out_flat, 1)

            non_pad_mask = (seqs_flat != model.pad_idx)
            correct = correct + ((predicted == seqs_flat) & non_pad_mask).sum().item()
            total = total + non_pad_mask.sum().item()

            all_predictions.extend(predicted[non_pad_mask].cpu().numpy())
            all_targets.extend(seqs_flat[non_pad_mask].cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = 100 * (correct / total) if total > 0 else 0

    return avg_loss, accuracy, all_predictions, all_targets
