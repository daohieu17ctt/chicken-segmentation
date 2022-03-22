from torch.utils.data import Dataset
import pandas as pd
import os 
import torch
from PIL import Image
import torchvision.transforms as transforms
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
        
class ChickenDataset(Dataset):

    def __init__(self, root_dir, num_class=2, transform=None):
        

    def __len__(self):
        return len(self.novel_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.novel_frame.iloc[idx, 0])
        sample = pil_loader(img_name)
      
        y = int(self.novel_frame.iloc[idx, 1])-1
        if(y<0):
            y= self.num_class
        if self.transform:
            sample = self.transform(sample)

        return sample,y