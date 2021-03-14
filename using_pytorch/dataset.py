import numpy as np 
import cv2 
import albumentations
import torch 



class custom_dataset():
    def __init__(self, image, label, train_data_aug=False):
        self.images = image
        self.labels = label
        
        if train_data_aug:
            self.aug = albumentations.Compose([
                                albumentations.Resize(256, 256, always_apply=True),
                                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                                scale_limit=0.1,
                                                                rotate_limit=5,
                                                                p=0.9),
                                #albumentations.RandomBrightnessContrast(always_apply=False),
                                albumentations.RandomRotate90(always_apply=False),
                                albumentations.HorizontalFlip(),
                                albumentations.VerticalFlip(),
                                albumentations.Normalize(mean=(0.485, 0.456, 0.406), 
                                                         std=(0.229, 0.224, 0.225), 
                                                         always_apply=True)              
                                                ])

        else:
            self.aug = albumentations.Compose([
                                albumentations.Resize(256, 256, always_apply=True),
                                albumentations.Normalize(mean=(0.485, 0.456, 0.406), 
                                                         std=(0.229, 0.224, 0.225),
                                                         always_apply=True) 
                                ]) 
        
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        image = cv2.imread(image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256)).astype(float)
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        label = self.labels[idx]
        
        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label 

