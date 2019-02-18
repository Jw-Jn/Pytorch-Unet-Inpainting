import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

class InpaintingDataSet(Dataset):

    def __init__(self, img_path, crop_num, train=True):
        
        self.img_path = img_path
        self.crop_size = 200
        self.img_patches = self.generatePatches(crop_num, self.crop_size)
        self.train = train

    def __len__(self):
        return len(self.img_patches)

    def __getitem__(self, idx):
        img = self.img_patches[idx]

        # img_origin = img

        # data augmentation
        if self.train:
            img = self.randomFlip(img)
            img = self.randomRotate(img)
            img = self.randomResize(img)
            img = self.randomCrop(img)
            img = self.colorJitter(img)

        resize = transforms.Resize((128, 128))
        img = resize(img)

        mask = self.generateMask(img)

        img = np.array(img, dtype=np.float) / 255.0
        # img_origin = np.array(img_origin, dtype=np.float) / 255.0

        ########## for test ##############
        # plt.subplot(1, 3, 1)
        # plt.imshow(img_origin)
        # plt.subplot(1, 3, 2)
        # plt.imshow(img)
        # plt.subplot(1, 3, 3)
        # plt.imshow(mask*255)
        # plt.show()

        img_tensor = torch.from_numpy(img).float()
        # mask_tensor = torch.from_numpy(mask).float()

        return img_tensor, mask

    
    def generateMask(self, img, hole_size=[8, 64], holes_num=5):

        img_h, img_w = img.size
        # mask = torch.zeros((img_h, img_w))
        mask = torch.ones((img_h, img_w, 1))
        for _ in range(holes_num):

            # choose patch size
            if random.random() < 0.5:
                hole_w = hole_size[0]
                hole_h = hole_size[1]
            else:
                hole_w = hole_size[1]
                hole_h = hole_size[0]

            # choose offset upper-left coordinate
            offset_x = random.randint(0, img_w - hole_w)
            offset_y = random.randint(0, img_h - hole_h)
            # mask[offset_y : offset_y + hole_h, offset_x : offset_x + hole_w, :] = 1.0
            mask[offset_y : offset_y + hole_h, offset_x : offset_x + hole_w, :] = 0.0

        return mask
    
    def generatePatches(self, patch_num, crop_size):
        patches = []
        img = Image.open(self.img_path)

        for _ in range(patch_num):
            crop = transforms.RandomCrop(crop_size)
            patch = crop(img)
            patches.append(patch)

        return patches

    def randomFlip(self, img):
        if random.random() < 0.5:
            # print('flip')
            if random.random() < 0.5:
                flip = transforms.RandomVerticalFlip()
            else:
                flip = transforms.RandomHorizontalFlip()

            img = flip(img)
        return img

    def randomRotate(self, img):
        if random.random() < 0.5:
            # print('rotate')
            degree = [90, 180, -90]
            img = transforms.functional.rotate(img, degree[random.randint(0,2)])
            # rotate = transforms.RandomRotation(180)
            # img = rotate(img)
        return img

    def colorJitter(self, img):
        if random.random() < 0.5:
            # print('cj')
            colorJ = transforms.ColorJitter()
            img = colorJ(img)
        return img

    def randomCrop(self, img):
        if random.random() < 0.5:
            # print('crop')
            crop = transforms.RandomCrop(150)
            img = crop(img)
        return img

    def randomResize(self, img):
        if random.random() < 0.5:
            # print('resize')
            w = random.randint(-50, 50)
            h = random.randint(-50, 50)
            resize = transforms.Resize((self.crop_size+w, self.crop_size+h))
            img = resize(img)
        return img