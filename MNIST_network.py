#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:52:32 2026

@author: blainebehen
"""

# Based on APML seminar lecture notes; rewritten as a learning exercise.

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------- Data ----------
train_data = datasets.MNIST(root="./data", train=True, download=True,
                            transform=transforms.ToTensor())
                            # 60,000 training images, each 1 x 28 x 28, pixel values 0-1

test_data = datasets.MNIST(root="./data", train=False, download=True,
                           transform=transforms.ToTensor())
                           # 10,000 test images, each 1 x 28 x 28, pixel values 0-1

loader = DataLoader(train_data, batch_size=64, shuffle=True)
                           # 938 batches per epoch (last one has 32 images)
                           # each batch: x is 64 x 1 x 28 x 28, y is 64

test_loader = DataLoader(test_data, batch_size=1000)
                           # 10 batches
                           # each batch: x is 1000 x 1 x 28 x 28, y is 1000

# ---------- Parameters ----------
V = (torch.randn(64, 784) * 0.01).requires_grad_(True)   # v_{j,i}: detector j's weight on pixel i
c = torch.zeros(64).requires_grad_(True)                 # c_j: detector j's intercept
W = (torch.randn(10, 64) * 0.01).requires_grad_(True)    # w_{k,j}: digit k's weight on detector j
b = torch.zeros(10).requires_grad_(True)                 # b_k: digit k's intercept

# ---------- Part 1: overfit one batch (sanity check) ----------
x, y = next(iter(loader))     # x = images, y = true digits
x = x.view(-1, 784)           # x_i for each image: 64 x 784

for step in range(201):
    t = x @ V.T + c           # t_j = sum_i v_{j,i} * x_i + c_j           (64 x 64)
    h = torch.relu(t)         # h_j = max(0, t_j)                          (64 x 64)
    s = h @ W.T + b           # s_k = sum_j w_{k,j} * h_j + b_k            (64 x 10)
    loss = F.cross_entropy(s, y)
                              # p_k = e^{s_k} / sum_m e^{s_m}
                              # loss = mean over batch of -log p_y
    loss.backward()           # computes d(loss)/d(theta) for theta in V, c, W, b

    with torch.no_grad():
        for p in [V, c, W, b]:
            p -= 0.1 * p.grad # theta <- theta - 0.1 * d(loss)/d(theta)
            p.grad.zero_()    # reset gradients (PyTorch adds them up otherwise)

    if step % 20 == 0:
        print("step:", step, "loss:", loss.item())

# ---------- Accuracy on unseen images ----------
def accuracy(data_loader):
    correct, total = 0, 0
    with torch.no_grad():                      # just predicting, no gradients needed
        for x, y in data_loader:               # x = images (1000 x 1 x 28 x 28), y = true digits (1000)
            x = x.view(-1, 784)                # x_i for each image                  (1000 x 784)
            t = x @ V.T + c                    # t_j = sum_i v_{j,i} * x_i + c_j     (1000 x 64)
            h = torch.relu(t)                  # h_j = max(0, t_j)                    (1000 x 64)
            s = h @ W.T + b                    # s_k = sum_j w_{k,j} * h_j + b_k      (1000 x 10)
            preds = s.argmax(dim=1)            # guess = digit k with the highest s_k (1000)
            correct += (preds == y).sum().item()   # count guesses that match the truth
            total += len(y)                    # count images seen (+1000 per batch)
    return correct / total                     # fraction correct

print("Test accuracy before full training:", accuracy(test_loader))

# ---------- Part 2: train on all 60,000 images ----------
for epoch in range(10):                         # epoch = one full pass through all 60,000 images
    for x, y in loader:                        # 938 batches: x is 64 x 1 x 28 x 28, y is 64
        x = x.view(-1, 784)                    # x_i for each image                  (64 x 784)
        t = x @ V.T + c                        # t_j = sum_i v_{j,i} * x_i + c_j     (64 x 64)
        h = torch.relu(t)                      # h_j = max(0, t_j)                    (64 x 64)
        s = h @ W.T + b                        # s_k = sum_j w_{k,j} * h_j + b_k      (64 x 10)
        loss = F.cross_entropy(s, y)           # loss = (1/64) * sum_n -log p_{y_n}   (one average)
        loss.backward()                        # computes d(loss)/d(theta) for theta in V, c, W, b

        with torch.no_grad():
            for p in [V, c, W, b]:
                p -= 0.1 * p.grad              # theta <- theta - 0.1 * d(loss)/d(theta)
                p.grad.zero_()                 # reset gradients

    print(f"Epoch {epoch}: last batch loss {loss.item():.3f}, "
          f"test accuracy {accuracy(test_loader):.3f}")