import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random
import numpy as np

class Resnet50(nn.Module):
    """
    Resnet50 model

    Args:
        num_class: number of output classes
    """
    def __init__(self, num_class=2):
        super().__init__()
        self.num_class = num_class
        self.model = models.resnet50()#dont use pretrained
        # Add more fc layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=self.num_class))


    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input image, shape (B, C, H, W)
        Returns:
            torch.Tensor: output logits tensor, shape (B, num_class)
        """
        out = self.model(x)
        return out