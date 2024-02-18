import enum
from enum import Enum
import torch
import PIL
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
class DatasetSplit(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
IMAGE_MEAN = [0.485]
IMAGENET_STD = [0.229]
class CMIDataset(torch.utils.data.Dataset):
    def __init__(self,
                 source,
                 resize = 60,
                 image_size = 65,
                 split = DatasetSplit.TRAIN,
                 train_val_split = 1.0,
                 rotate_degree = 0,
                 translate = 0,
                 brightness = 0,
                 contrast = 0,
                 h_flip_p = 0,
                 v_flip_p = 0,
                 scale = 0,
                 **kwargs):
        super().__init__()
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGE_MEAN
        self.source = source
        self.split = split
        self.train_val_split = train_val_split
        self.classnames_to_use = ['CMI']
        self.transform_img = [
            transforms.Resize(resize),
            transforms.RandomRotation(rotate_degree),
            transforms.RandomAffine(translate, scale=(1-scale, 1+scale)),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.ColorJitter(brightness, contrast),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]

        self.transform_img = transforms.Compose(self.transform_img)
        
        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.image_size = (1, image_size, image_size)
        self.imagepaths_per_class, self.data_to_iterate = self.get_image_data()
    def __len__(self):
        return len(self.data_to_iterate)
    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = np.load(image_path)
        image = PIL.Image.fromarray(image)
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = np.load(mask_path).astype(np.uint8)
            mask = PIL.Image.fromarray(mask)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])
        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != 'good'),
            # "image_name": "/".join(image_path.split('/')[-4:]),
            "image_path": image_path,
        }
    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, 'ground_truth')
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, file) for file in anomaly_files
                ]
                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[classname][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[classname][anomaly][train_val_split_idx:]
                
                if self.split == DatasetSplit.TEST and anomaly != 'good':
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, file) for file in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname][anomaly] = [None]
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)
        return imgpaths_per_class, data_to_iterate

if __name__ == '__main__':
    dataset = CMIDataset('/Users/tranthanh/Downloads/create_dataset/Dataset')
    print(dataset[0]['mask'])
    plt.imshow(dataset[0]['mask'][0])
    plt.show()
    # img = np.load('/Users/tranthanh/Downloads/create_dataset/Dataset/a/train/good/384.npy')
    # print(img.size())