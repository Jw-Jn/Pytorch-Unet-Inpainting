import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class InpaintingDataSet(Dataset):

    def __init__(self, img_path):
        
        self.img_path = img_path
        self.img_patches = self.generate_patches(50)

    def __len__(self):
        return len(self.img_patches)

    def __getitem__(self, idx):
        img = self.img_patches[idx]

        # data augmentation
        img = self.random_flip(img)
        img = self.random_rotate(img)
        img = self.random_resize(img)
        img = self.random_crop(img)
        img = self.color_jitter(img)

        resize = transforms.Resize((128, 128))
        img = resize(img)

        mask = self.generate_mask(img)
        
        img = np.array(img, dtype=np.float) / 255.0 * 2 - 1

        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).float()

        return img_tensor, mask_tensor

    
    def generate_mask(self, img, hole_size=[8, 64], holes_num=5):

        img_h, img_w = img.shape
        mask = torch.zeros((img_h, img_w))
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
            mask[offset_y : offset_y + hole_h, offset_x : offset_x + hole_w] = 1.0

        return mask
    
    def generate_patches(self, patch_num):
        patches = []
        img = Image.open(self.img_path)

        for _ in range(patch_num):
            crop = transforms.RandomCrop(20)
            patch = crop(img)
            patches.append(patch)

        return patches

    def random_flip(self, img):
        if random.random() < 0.5:
            if random.random < 0.5:
                flip = transforms.RandomVerticalFlip()
            else:
                flip = transforms.RandomHorizontalFlip()

            img = flip(img)
        return img

    def random_rotate(self, img):
        degree = [90, 180, -90]
        if random.random() < 0.5:
            rotate = transforms.RandomRotation(degree[random.randint(0,2)])
            img = rotate(img)
        return img

    def color_jitter(self, img):
        if random.random() < 0.5:
            colorJ = transforms.ColorJitter()
            img = colorJ(img)
        return img

    def random_crop(self, img):
        if random.random() < 0.5:
            crop = transforms.RandomCrop(15)
            img = crop(img)
        return img

    ############
    def random_resize(self, img):
        if random.random() < 0.5:
            resize = transforms.RandomResizedCrop()
            img = resize(img)
        return img



# get list
# def get_list(file_path):
#     data_list = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             if not line in ['\n', '\r\n']:
#                 result = line.split('\t')
#                 img_name = result[0]
#                 bbox_list = result[1].split()
#                 bbox = np.array(bbox_list).astype(np.int).reshape((-1,2))
#                 landmarks_list = result[2].split()
#                 landmarks = np.array(landmarks_list).astype(np.float).reshape((-1,2))
#                 data_list.append({'name': img_name, 'bbox': bbox, 'lm': landmarks})
#     return data_list

# def show_landmarks(img, landmarks):
#     length = len(img)
#     figs, axes = plt.subplots(1, length)
#     for i in range(0, length):
#         axes[i].imshow((img[i] + 1) / 2)
#         axes[i].set_title('Sample:' + str(i))
#         axes[i].scatter(landmarks[i][0]*img_size, landmarks[i][1]*img_size, s=20, marker='.', c='r')
#     plt.pause(0.01)  # pause a bit so that plots are updated
#     plt.show()


# img_size = 128
# random_crop_r = 5
# lfw_dir = '../../../../Courses_data'
# lfw_dataset_dir = '../../../../Courses_data/lfw'
# train_set_path = os.path.join(lfw_dir, 'LFW_annotation_train.txt')
# test_set_path = os.path.join(lfw_dir, 'LFW_annotation_test.txt')

# test for original img and landmarks
# test_list = test_list[0: 4]
# figs, axes = plt.subplots(1, 4)
# for i in range(0, 4):
#     item = test_list[i]
#     img_name = item['name']
#     bbox = item['bbox']
#     landmarks = item['lm']
#     img_path = img_name.split('.')[0][0:-5]
#     file_path = os.path.join(lfw_dataset_dir, img_path, img_name)
#     img = np.array(Image.open(file_path), dtype=np.float) / 255
#     axes[i].imshow(img)
#     landmarks = np.array(landmarks).T
#     axes[i].scatter(landmarks[0], landmarks[1], s=20, marker='.', c='r')
#     axes[i].set_title('Sample:' + str(i))
# plt.pause(0.01)
# plt.show()