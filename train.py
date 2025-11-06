import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        out = model(data)

        loss = criterion(out,target)

        loss.backward()

        optimizer.step()

        total_loss = total_loss + loss.item() * data.size(0)

        _, predicted = torch.max(out, 1)

        correct = correct + (predicted == target).sum().item()

        total = total + target.size(0)

        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    if scheduler is not None:

      try:
          scheduler.step()

      except ZeroDivisionError:

          print("T_max is zero")

    avg_loss = total_loss/total

    accuracy = 100 * (correct/total)

    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for data, target in val_loader:

            data, target = data.to(device), target.to(device)

            out = model(data)

            loss = criterion(out,target)

            total_loss = total_loss + loss.item() * data.size(0)

            _, predicted = torch.max(out, 1)

            correct = correct + (predicted == target).sum().item()

            total = total + target.size(0)

            all_predictions.extend(predicted.cpu().numpy())

            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss/total

    accuracy = 100 * (correct/total)

    return avg_loss, accuracy, all_predictions, all_targets

