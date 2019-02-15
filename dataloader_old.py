import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1, img_size=388, desire_size=572,
                flip=False, rotate=False, gamma_c=False, zoom=False, deform=False):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

        self.img_size = img_size
        self.desire_size = desire_size

        self.flip = flip
        self.rotate = rotate
        self.gamma_c = gamma_c
        self.zoom = zoom
        self.deform = deform

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        index = [i for i in range(current, endId)]
        if self.mode == 'train':
            random.shuffle(index)

        # while current < endId:
        for current in index:
            # todo: load images and labels
            # hint: scale images between 0 and 1

            data_image = Image.open(self.data_files[current])
            label_image = Image.open(self.label_files[current])

            # data augment
            if self.mode == 'train':
                if self.flip:
                    data_image, label_image = self.random_flip(data_image, label_image)
                if self.rotate:
                    data_image, label_image = self.random_rotate(data_image, label_image)
                if self.zoom:
                    data_image, label_image = self.random_zoom(data_image, label_image)
                if self.gamma_c:    
                    data_image = self.gamma_correction(data_image)

            data_image = data_image.resize((self.img_size, self.img_size))
            label_image = label_image.resize((self.img_size, self.img_size))

            data_image = np.array(data_image, dtype=np.float32) #/ 255.0 * 2 - 1
            label_image = np.array(label_image, dtype=np.int)

            # data augment
            if self.mode == 'train':
                if self.deform:
                    data_image, label_image = self.elastic_transform(data_image, label_image)         

            data_image = data_image / 255.0 * 2 - 1
            padding = (self.desire_size - self.img_size)//2
            data_image = np.pad(data_image, padding, 'symmetric')

            # current += 1
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def random_flip(self, img, label):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label

    def random_rotate(self, img, label):
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            label = label.transpose(Image.ROTATE_90)
        return img, label

    def random_zoom(self, img, label):
        if random.random() < 0.5:
            a = random.randint(1, 10)
            box = (a, a, img.size[0]-a, img.size[1]-a)
            img = img.crop(box)
            label = label.crop(box)
        return img, label

    def gamma_correction(self, img, gamma=random.uniform(2,4)):
        if random.random() < 0.5:
            img = transforms.functional.adjust_gamma(img, gamma)
        return img

    def elastic_transform(self, img, label, alpha=30, sigma=5, random_state=None):
        if random.random() < 0.5:
            if random_state is None:
                random_state = np.random.RandomState(None)

            shape = img.shape

            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

            img = map_coordinates(img, indices, order=1).reshape(shape)
            label = map_coordinates(label, indices, order=1).reshape(shape)
        
        return img, label