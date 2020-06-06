"""Utilities for training
"""
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST


def add_noise(img):
    """Add noise to input image

    Args:
        img (torch tensor): input image

    Returns:
        torch tensor: noisy image
    """
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def load_mnist():
    # the data, shuffled and split between train and test sets
    train = MNIST(
        root="./data/",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    test = MNIST(root="./data/", train=False, transform=transforms.ToTensor())
    x_train, y_train = train.train_data, train.train_labels
    x_test, y_test = test.test_data, test.test_labels
    x = torch.cat((x_train, x_test), 0)
    y = torch.cat((y_train, y_test), 0)
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.0)
    print("MNIST samples", x.shape)
    return x, y
