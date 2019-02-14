import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            # todo: load images and labels
            # hint: scale images between 0 and 1

            data_image = Image.open(self.data_files[current]) # make it grey scale??
            data_image = np.array(data_image, dtype=np.float) / 255.0 * 2 - 1

            label_image = Image.open(self.label_files[current])
            label_image = np.array(label_image, dtype=np.int)

            # hint: if training takes too long or memory overflow, reduce image size!
            
            # data_image = data_image.resize((img_size, img_size))
            # label_image = label_image.resize((img_size, img_size))

            # data_image = torch.from_numpy(img).float()
            # label_image = torch.from_numpy(label).float()

            current += 1
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))