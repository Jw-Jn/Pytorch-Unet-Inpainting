import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import transforms

import matplotlib.pyplot as plt
# from PIL import Image, ImageOps

from model import UNet
from dataloader import DataLoader

import time

def train_net(net,
              epochs=5,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              gpu=False,
              train=True,
              pth_dir='./data/cells/checkpoints/CP10.pth',
              flip=False,
              rotate=False,
              zoom=False,
              gamma_c=False,
              deform=False):

    loader = DataLoader(data_dir, flip=flip, rotate=rotate, gamma_c=gamma_c, zoom=zoom, deform=deform)

    N_train = loader.n_train()
 
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.99, weight_decay=0.0005)
    
    if train:
        loader.setMode('train')

        for epoch in range(epochs):
            print('Epoch %d/%d' % (epoch + 1, epochs))
            print('Training...')
            net.train()

            epoch_loss = 0

            start = time.process_time()

            for i, (img, label) in enumerate(loader):
                shape = img.shape
                label = label - 1
                # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
                # img = Image.fromarray(img*255.)
                # img.show()
                
                img = torch.from_numpy(img)
                img = img.view(1, 1, shape[0], shape[1])

                # todo: load image tensor to gpu
                if gpu:
                    img = Variable(img.cuda())

                # todo: get prediction and getLoss()
                pred_label = net(img)

                loss = getLoss(pred_label, label)

                epoch_loss += loss.item()
    
                # print('Training sample %d / %d - Loss: %.6f' % (i+1, N_train, loss.item()))

                # optimize weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch+1)==40 or (epoch+1)==50 or (epoch+1)==60 or (epoch+1)==70:
                torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
                print('Checkpoint %d saved !' % (epoch + 1))
            print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))
            print('Time: ', time.process_time() - start)

    
    
    # displays test images with original and predicted masks after training
    else:
        print('Testing')
        loader.setMode('test')
        net_state = torch.load(pth_dir)
        net.load_state_dict(net_state)
        net.eval()
        with torch.no_grad():
            for i, (img, label) in enumerate(loader):
                shape = img.shape
                img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
                if gpu:
                    img_torch = img_torch.cuda()
                pred = net(img_torch)
                pred_sm = softmax(pred)
                _,pred_label = torch.max(pred_sm,1)

                pad = (572-388)//2
                img = img[pad:572-pad, pad:572-pad]
                plt.subplot(1, 3, 1)
                plt.imshow(img*255.)
                plt.subplot(1, 3, 2)
                plt.imshow((label-1)*255.)
                plt.subplot(1, 3, 3)

                plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
                # plt.show()
                plt.savefig('test_'+str(i)+'.png')

                # img = Image.fromarray(img*255.)
                # img.show()
                # label = Image.fromarray((label-1)*255.)
                # label.show()
                # pred_label = pred_label.cpu().detach().numpy().squeeze()*255.
                # pred_label = Image.fromarray(pred_label*255.)
                # pred_label.show()
            

def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    return cross_entropy(p, target_label)

def softmax(input):
    # todo: implement softmax function
    p = torch.exp(input.float()) / torch.sum(torch.exp(input.float()), dim=1)
    # print(p)
    return p

def cross_entropy(input, targets):
    # todo: implement cross entropy
    # Hint: use the choose function
    pred = choose(input, targets)
    pixel = pred.size()[0]*pred.size()[1]
    ce = -torch.sum(torch.log(pred)) / pixel
    return ce

# Workaround to use numpy.choose() with PyTorch
def choose(pred_label, true_labels):
    size = pred_label.size()   # (N,C,H,W)
    ind = np.empty([size[2]*size[3],3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i,:] = [true_labels[x,y], x, y]
            i += 1

    pred = pred_label[0, ind[:,0], ind[:,1], ind[:,2]].view(size[2],size[3])

    return pred
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('--test', action='store_false', default=True, help='testing mode')
    parser.add_option('--pth', default='./data/cells/checkpoints/CP60.pth', help='pth directory')
    parser.add_option('--frz', action='store_true', default=False, help='flip rotate zoom')
    parser.add_option('--gamma', action='store_true', default=False, help='gamma correction')
    parser.add_option('--deform', action='store_true', default=False, help='deformation')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes, bn=True)

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        n_classes=args.n_classes,
        gpu=args.gpu,
        data_dir=args.data_dir,
        train=args.test,
        pth_dir=args.pth,
        flip=args.frz,
        rotate=args.frz,
        zoom=args.frz,
        gamma_c=args.gamma,
        deform=args.deform)

