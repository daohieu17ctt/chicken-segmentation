from posixpath import split
from tkinter import image_names
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder, ImageNet
from PIL import Image

from general_lightning_model import ChickenClassification
from backbone import *
import torch
import os
import albumentations as A
import cv2


class TestDataset(Dataset):
    def __init__(self, root_dir, transform) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.img_list = os.listdir(root_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.img_list[idx])
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return self.img_list[idx], img

# SEED= 2022
# pl.seed_everything(SEED)
transform_dict = {
    'train':    transforms.Compose(
        [   transforms.Resize((244,244)),
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


data_set = TestDataset("Dataset/test", transform=transform_dict['val'])
test_loader = DataLoader(data_set, batch_size=64)
# print(data_set[0][0].size())
# split_ratio = 0.8
# train_len = int(len(data_set) * split_ratio)
# val_len = len(data_set) - train_len
# train_set, val_set = torch.utils.data.random_split(data_set, [train_len, val_len])

# train_loader = DataLoader(train_set,
#                         batch_size=64,
#                         shuffle=True,num_workers=8)

# val_loader = DataLoader(val_set,
#                         batch_size=64,
#                         shuffle=False,num_workers=8)


# model = ChickenClassification(backbone)
# checkpoint_callback = ModelCheckpoint(monitor='val_loss',mode='min', save_top_k=5)
# # checkpoint_path= 'output/classification_model.pth'
# checkpoint_path = None
# trainer = pl.Trainer(max_epochs=10, checkpoint_callback=checkpoint_callback, resume_from_checkpoint=checkpoint_path)
# trainer.fit(model, train_loader, val_loader)

backbone = Resnet50(num_class=2)
model = ChickenClassification.load_from_checkpoint("output/epoch=9-step=69.ckpt", model=backbone)
model.model.eval()

with open("classification.txt", 'w') as f:
    for batch_idx, batch in enumerate(test_loader):
        # print(batch_idx)
        img_name, img = batch
        img_name = np.array(img_name)
        logits = model.model(img)
        preds = torch.argmax(logits, dim=1).numpy()
        for ig, p in zip(img_name, preds):
            f.write("{}   {}\n".format(ig, p))
