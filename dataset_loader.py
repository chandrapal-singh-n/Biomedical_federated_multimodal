import os
import torch
import torchvision
from torch.utils.data import Dataset
import PIL
from PIL import Image

class CustomDatasetLoader(Dataset):
    
    def __init__(self, root_dir,transform=None):
        self.root_dir = root_dir
        self.images = []
        self.labels =[]
        self.transform = transform
        
        self.class_to_idx = {}
        
        self.classes = sorted(os.listdir(root_dir))
        
        for idx,class_names in enumerate(self.classes):
            self.class_to_idx[class_names]=idx
        
        for class_names in self.classes:
            class_folder = os.path.join(root_dir,class_names)
            image_names = os.listdir(class_folder)
            
            for image_name in image_names:
                img_path = os.path.join(class_folder,image_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_names])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img =Image.open(img_name).convert("RGB")
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)

        return img,label


# class CustomBrainImageDataset(Dataset):
    
#     def __init__(self, root_dir, transform=None):
#         self.root_dir= root_dir
#         self.image_path = []
#         self.labels =[]
#         self.transform= transform
        
#         self.class_to_idx={}
        
#         self.classes= sorted(os.listdir(root_dir))
        
#         for idx,class_name in enumerate(self.classes):
#             self.class_to_idx[class_name]=idx
#         print(self.class_to_idx)
            
                   
#         for class_name in self.classes:
#             class_folder = os.path.join(root_dir,class_name)
#             image_names = os.listdir(class_folder)
            
#             for image_name in image_names:
#                 img_path = os.path.join(class_folder,image_name)
#                 self.image_path.append(img_path)
#                 self.labels.append(self.class_to_idx[class_name])
#         print(f"total images {len(self.image_path)}")
#         print(f"total number of classes {len(self.classes)}")
        
    
#     def __len__(self):
#         return len(self.image_path)
    
#     def __getitem__(self, index):
        
#         img_name= self.image_path[index]
#         img = Image.open(img_name)
#         img_label = self.labels[index]
        
#         if self.transform:
#             img = self.transform(img)
#         return img, img_label