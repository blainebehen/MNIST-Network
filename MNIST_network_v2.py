#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 23:05:36 2026

@author: blainebehen
"""

# Based on APML seminar lecture notes; rewritten as a learning exercise.
# Version 2: same network, built with nn.Module and an optimizer.

import torch
import torch.nn as nn
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

# ---------- Model ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.detectors = nn.Linear(784, 64)    # holds V (64 x 784) and c (64): v_{j,i}, c_j
        self.scores = nn.Linear(64, 10)        # holds W (10 x 64) and b (10):  w_{k,j}, b_k

    def forward(self, x):                      # x: B x 1 x 28 x 28   (B = batch size)
        x = x.view(-1, 784)                    # x_i for each image                  (B x 784)
        t = self.detectors(x)                  # t_j = sum_i v_{j,i} * x_i + c_j     (B x 64)
        h = torch.relu(t)                      # h_j = max(0, t_j)                    (B x 64)
        s = self.scores(h)                     # s_k = sum_j w_{k,j} * h_j + b_k      (B x 10)
        return s

model = Net()                                  # creates all parameters: V, c, W, b
criterion = nn.CrossEntropyLoss()              # p_k = e^{s_k} / sum_m e^{s_m};  loss = mean of -log p_y
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                                               # update rule: theta <- theta - 0.1 * d(loss)/d(theta)

# ---------- Part 1: overfit one batch (sanity check) ----------
x, y = next(iter(loader))                      # x = images (64 x 1 x 28 x 28), y = true digits (64)

for step in range(201):
    s = model(x)                               # forward pass: x -> t -> h -> s      (64 x 10)
    loss = criterion(s, y)                     # loss = (1/64) * sum_n -log p_{y_n}
    optimizer.zero_grad()                      # reset gradients (replaces p.grad.zero_())
    loss.backward()                            # computes d(loss)/d(theta) for every parameter
    optimizer.step()                           # theta <- theta - 0.1 * d(loss)/d(theta)

    if step % 20 == 0:
        print("step:", step, "loss:", loss.item())

# ---------- Accuracy on unseen images ----------
def accuracy(data_loader):
    correct, total = 0, 0
    with torch.no_grad():                      # just predicting, no gradients needed
        for x, y in data_loader:               # x = images (1000 x 1 x 28 x 28), y = true digits (1000)
            s = model(x)                       # scores for each image               (1000 x 10)
            preds = s.argmax(dim=1)            # guess = digit k with the highest s_k (1000)
            correct += (preds == y).sum().item()   # count guesses that match the truth
            total += len(y)                    # count images seen (+1000 per batch)
    return correct / total                     # fraction correct

print("Test accuracy before full training:", accuracy(test_loader))

# ---------- Part 2: train on all 60,000 images ----------
for epoch in range(10):                        # epoch = one full pass through all 60,000 images
    for x, y in loader:                        # 938 batches: x is 64 x 1 x 28 x 28, y is 64
        s = model(x)                           # forward pass                        (64 x 10)
        loss = criterion(s, y)                 # loss = (1/64) * sum_n -log p_{y_n}
        optimizer.zero_grad()                  # reset gradients
        loss.backward()                        # computes d(loss)/d(theta) for every parameter
        optimizer.step()                       # theta <- theta - 0.1 * d(loss)/d(theta)

    print(f"Epoch {epoch}: last batch loss {loss.item():.3f}, "
          f"test accuracy {accuracy(test_loader):.3f}")