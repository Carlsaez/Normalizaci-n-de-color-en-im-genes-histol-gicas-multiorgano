import os
from PIL import Image, ImageOps
import torchvision.transforms.functional as fn
import numpy as np

class ImageDataset():
    def __init__(self, dir_data, data_group, transform=None):
        self.dir_data = dir_data
        self.data_group = data_group
        self.files = self.list_dir_files()
        self.length = len(self.files)
        self.transform = transform
        
    def __len__(self):
        return self.length

    def list_dir_files(self):
        list_files = []
        for root, dir, files in os.walk(self.dir_data):
            for file in files:
                if not file.endswith("DS_Store"):
                    list_files.append(os.path.join(root, file))
            list_files = list(filter(lambda element: self.data_group in element, list_files))
        return list_files

    def __getitem__(self, idx):
        length = self.__len__()
        img_path = self.files[idx % length]
        img = (Image.open(img_path).convert("RGB"))
        if 'BreastCancer' in img_path: 
            size = (img.size[0]//4, img.size[1]//4)
            img = ImageOps.crop(img, size)
        if self.transform:
            augmented = self.transform(image=np.array(img)) 
            img = augmented['image']
        elif img.size[0] != 256 or img.size[1] != 256:
            img = fn.resize(img, size=[256, 256])
        img = fn.to_tensor(img)
        return img