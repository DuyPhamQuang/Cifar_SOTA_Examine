#!/usr/bin/env python
# coding: utf-8

# # Deep Residual Learning for Image Recognition
# This notebook provides a PyTorch implementation of [Deep Residual Learning for Image Recognition" (He et al., 2015)](https://arxiv.org/pdf/1512.03385)

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

from models.resnet import ResNet
from data_loader import get_data_loaders, plot_images
from trainer import train_one_epoch, evaluate
from utils import save_checkpoint, load_checkpoint, plot_history


# ## Data Augmentation, Preprocess (Normalization)

# In[ ]:


# data directory
data_dir = 'data/cifar10'
batch_size = 128


# In[ ]:


# =======================================================================================
# DATA LOADING (Before Augementation For Visualizing Purpose)
# =======================================================================================
train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

# ---Load data---
train_loader, test_loader = get_data_loaders(data_dir,
                                             batch_size,
                                             train_transform,
                                             test_transform,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True)


# In[ ]:


# ---Visualize---
# Training images
data_iter = iter(train_loader)
images, labels = next(data_iter)
X = images.numpy().transpose([0, 2, 3, 1])
plot_images(X, labels)
print(images.shape)


# In[ ]:


# =======================================================================================
# DATA AUGMENTATION AND NORMALIZATION
# =======================================================================================

# Normalization variable
# The normalization values are calculated from the training set and are used to normalize both the training and testing sets.
# They have been pre_computed publicly and are commonly used for the CIFAR-10 dataset in practice.
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],   
                                 std=[0.2023, 0.1994, 0.2010])

# Training Set
train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize
])

# Testing Set
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


# In[ ]:


# ---Reload data with augmentation---
train_loader, test_loader = get_data_loaders(data_dir,
                                             batch_size,
                                             train_transform,
                                             test_transform,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True)


# In[ ]:


# Training set
data_iter = iter(train_loader)
images, labels = next(data_iter)
X = images.numpy().transpose([0, 2, 3, 1])
plot_images(X, labels)
print(images.shape)


# In[ ]:


# Testing set
data_iter = iter(test_loader)
images, labels = next(data_iter)
X = images.numpy().transpose([0, 2, 3, 1])
plot_images(X, labels)
print(images.shape)


# ## Training

# In[ ]:


# --- Hyperparameters (according to section 4.2 the paper) ---
BATCH_SIZE   = 128

# Optimizer
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4

# Learning-rate schedule
# 50,000 training images / batch size 128 ≈ 391 iterations per epoch
# 64k iterations / 391 ≈ 164 epochs
EPOCHS       = 164
# 32k iterations/391 ≈ 82 epochs, 50k iterations/391 ≈ 123 epochs
MILESTONES   = [82, 123]
GAMMA        = 0.1


# In[ ]:


# --- Training/Evaluating process ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# n determines network size (depth = 6n + 2)
# Resnet-20, 32, 44, 56
ns = [3, 5, 7, 9]

# For mounting to Google Drive when runnung on COLAB - comment if not the case
#results_path = '/content/drive/MyDrive/Hiwi_TNT/ResNet_Training_2026'

results_path = './results'  # Local path for saving results (checkpoints, training history, plots)
os.makedirs(results_path, exist_ok=True)

# Iterate over different ResNet depths and train/evaluate each model
for n in ns:
    # if training_curve already exists, skip training
    training_curve_path = os.path.join(results_path, f'training_curves_Resnet_{6*n + 2}.png')
    if os.path.exists(training_curve_path):
        print(f"Training curve for Resnet-{6*n + 2} already exists. Skipping training.")
        continue

    print(f'MODEL: Resnet-{6*n + 2}')
    model = ResNet(n, shortcuts=True).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr = LR,
        momentum = MOMENTUM,
        weight_decay = WEIGHT_DECAY,
        nesterov = True
    )

    # Scheduler - Multiplies LR by GAMMA at each epoch in MILESTONES
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=MILESTONES,
        gamma=GAMMA
    )

    # History (for plotting)
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': [],
    }

    best_test_acc = 0.0
    checkpoint_path = os.path.join(results_path, f'best_model_Resnet_{6*n + 2}.pth')

    # Training/Evaluating Loop
    print(f"Starting training — {EPOCHS} epochs\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        print(
            f"Epoch [{epoch:3d}/{EPOCHS}] | LR: {current_lr:.4f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        # Save record to csv file
        record_df = pd.DataFrame(history)
        record_df.to_csv(os.path.join(results_path, f'training_history_Resnet_{6*n + 2}.csv'), index=False)

        # Save best checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                },
                save_path=checkpoint_path
            )

    # Plot training curves
    print(f"\nTraining complete.")
    print(f"Best test accuracy : {best_test_acc:.2f}%")
    print(f"Best checkpoint    : {checkpoint_path}\n")

    plot_history(
        history,
        milestones=MILESTONES,
        save_path = os.path.join(results_path, f'training_curves_Resnet_{6*n + 2}.png') 
    )

