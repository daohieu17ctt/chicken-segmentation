from posixpath import split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder, ImageNet

from general_lightning_model import ChickenClassification
from backbone import *
import torch
import os
import albumentations as A

# SEED= 2022
# pl.seed_everything(SEED)
transform_dict = {
    'train':    transforms.Compose(
        [   transforms.Resize((960,960)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
    ),
    'val':      transforms.Compose(
        [   transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),]
    )}


data_set = ImageFolder("Dataset/sample_classification", transform=transform_dict['val'])
# print(data_set[0])
split_ratio = 0.8
train_len = int(len(data_set) * split_ratio)
val_len = len(data_set) - train_len
train_set, val_set = torch.utils.data.random_split(data_set, [train_len, val_len])

train_loader = DataLoader(train_set,
                        batch_size=64,
                        shuffle=True,num_workers=8)

val_loader = DataLoader(val_set,
                        batch_size=64,
                        shuffle=False,num_workers=8)

backbone = Resnet50(num_class=2)

model = ChickenClassification(backbone)
checkpoint_callback = ModelCheckpoint(monitor='val_loss',mode='min', save_top_k=5)
# checkpoint_path= 'output/classification_model.pth'
checkpoint_path = None
trainer = pl.Trainer(max_epochs=10, checkpoint_callback=checkpoint_callback, resume_from_checkpoint=checkpoint_path)
trainer.fit(model, train_loader, val_loader)
