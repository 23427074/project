from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from skimage import io
from torch.utils.data import Dataset
import random
import numpy as np


class Images_Dataset(Dataset):
    """Class for getting data as a Dict
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        sample : Dict of images and labels"""

    def __init__(self, images_dir, labels_dir, transformI = None, transformM = None):

        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.transformI = transformI
        self.transformM = transformM

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):

        for i in range(len(self.images_dir)):
            image = io.imread(self.images_dir[i])
            label = io.imread(self.labels_dir[i])
            if self.transformI:
                image = self.transformI(image)
            if self.transformM:
                label = self.transformM(label)
            sample = {'images': image, 'labels': label}

        return sample


def make_dataset(images_dir, labels_dir):
        samples = []
        #print(os.listdir(images_dir))
        #print(os.listdir(labels_dir))
        images  = os.listdir(images_dir)
        for i in images:
            img = images_dir + '/' + i
            # print(img)
            label = labels_dir + '/' + i
            samples.append((img, label))
           # samples.append(img)
           # samples.append(label)
            # print(samples)
        # print(i
        #print(len(samples))
        return samples


'''class Labels_Dataset_folder(torch.utils.data.Dataset):
    def __init__(self, labels_dir,transformM = None):
        self.labels = sorted(os.listdir(labels_dir))
        self.labels_dir = labels_dir
        self.transformM = transformM
        # self.samples = make_dataset(self.images_dir, self.labels_dir)
        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                        #torchvision.transforms.Resize((128,128)),
                        torchvision.transforms.CenterCrop(256),
                        torchvision.transforms.RandomRotation((-10,10)),
                        torchvision.transforms.Grayscale(),
                        # torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        #torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
                        ])
    def __len__(self): 
        print('num_data: ', len(self.labels))
        return len(self.labels)

    def __getitem__(self, i):
        #print('labels: ', i, self.labels[i])
        labels = self.labels_dir + '/' + self.labels[i]
        #print(labels)
        l1 = Image.open(labels)

        seed=np.random.randint(0,2**32) # make a seed with numpy generator

                                                       # print('img_size: ',len(img))

                                                               # apply this seed to target/label tranfsorms
        random.seed(seed)
        torch.manual_seed(seed)
        label = self.lx(l1)
                # print('label_size: ',len(label))


        return label '''




class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

        # img_size = 0
        # label_size = 0

    def __init__(self, images_dir, labels_dir, transformI = None, transformM = None):
        #self.images = sorted(os.listdir(images_dir))
        # self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM
        self.samples = make_dataset(self.images_dir, self.labels_dir)
        # for x in range(len(self.samples)):
            # print(self.samples[x])

        if self.transformI:
            self.tx = self.transformI
            #print('true')
        else:
            self.tx = torchvision.transforms.Compose([
                #torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop(256),
                #torchvision.transforms.RandomRotation((-5,5)),
                #torchvision.transforms.RandomHorizontalFlip(),
                #torchvision.transforms.RandomVerticalFlip(),
                #torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                # torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            # print('false')

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                #torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop(256),
                #torchvision.transforms.RandomRotation((-10,10)),
                torchvision.transforms.Grayscale(),
                # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])
            # print('false')

    def __len__(self):
        print('num_data: ', len(self.samples))
        return len(self.samples)

    def __getitem__(self, i):
        #print(i)
        # print('images: ', i , self.images[i])
        #for x in range(len(self.samples)):
        #print(i)
        #print(self.samples[i])
        #print(len(self.samples))
        #print(type(self.samples))
        images, labels = self.samples[i]
        #print(len(images))
        #print(len(labels))
        #images = self.images_dir + '/' + self.images[i]
        # print('lock')
        # print('labels:', i , self.labels[i])
        # for x in range(len(self.images)):
            # print(self.images[x])
       # print('image: ',i)
        # labels = self.labels_dir + '/' + self.labels[i]
        #print(images)
        i1 = Image.open(images)
        # print(type(i1))
        #print(labels)
        l1 = Image.open(labels)

        seed=np.random.randint(0,2**32) # make a seed with numpy generator

        # apply this seed to img tranfsorms
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.tx(i1)
        #print(type(img))
       # print('img_size: ',len(img))
        
        # apply this seed to target/label tranfsorms  
        random.seed(seed) 
        torch.manual_seed(seed)
        label = self.lx(l1)
       # print('label_size: ',len(label))
        

        return img, label

