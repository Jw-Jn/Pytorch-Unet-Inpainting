import torch
import numpy as np
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from optparse import OptionParser
from model import UNet
import os
import dataset
import random

def trainNet(net, data_dir, sample_dir, cpt_dir, epochs=100, gpu=True, train=True, pth=None):

    criterion = torch.nn.MSELoss()
    if gpu:
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    if train:
        train_dataset = dataset.InpaintingDataSet(os.path.join(data_dir, 'train.png'), 1600)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=16,
                                                        shuffle=False,
                                                        num_workers=0)
        print('train items:', len(train_dataset))

        for epoch in range(0, epochs):
            print('Epoch %d/%d' % (epoch + 1, epochs))
            print('Training...')
            net.train()

            epoch_loss = 0

            for i, (img, mask) in enumerate(train_data_loader):
            
                optimizer.zero_grad()

                img_input = torch.cat([img, mask], dim=-1)
                img_input = torch.transpose(img_input, 1, 3)

                if gpu:
                    img_input = Variable(img_input.cuda())
                    img = Variable(img.cuda())

                out = net.forward(img_input)

                loss = criterion(out, torch.transpose(img, 1, 3))

                epoch_loss += loss.item()
                
                loss.backward()
                optimizer.step()

                print('Training sample %d / %d - Loss: %.6f' % (i+1, 100, loss.item()))
                
            print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / (i+1)))

            idx = random.randint(0, 15) # batch size -1
            showSample(img[idx], mask[idx], torch.transpose(out, 1, 3)[idx], (epoch+1), sample_dir, train=True)

            if (epoch+1) == 1 or (epoch+1) == 5 or (epoch+1) == 10 or (epoch+1) == 50 or (epoch+1) == 100:
                torch.save(net.state_dict(), os.path.join(cpt_dir, 'CP%d.pth' % (epoch + 1)))
                print('Checkpoint %d saved !' % (epoch + 1))

    else:
        test_dataset = dataset.InpaintingDataSet(os.path.join(data_dir, 'test.png'), 10)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)
        print('test items:', len(test_dataset))

        print('Testing')
        net.load_state_dict(torch.load(os.path.join(cpt_dir, pth)))
        net.eval()

        with torch.no_grad():
            for i, (img, mask) in enumerate(test_data_loader):

                img_input = torch.cat([img, mask], dim=-1)
                img_input = torch.transpose(img_input, 1, 3)

                if gpu:
                    img_input = Variable(img_input.cuda())

                out = net.forward(img_input)

                showSample(img[0], mask[0], torch.transpose(out, 1, 3)[0], i, sample_dir, train=False)


def getArgs():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('--data-dir', dest='data_dir', default='./inpainting_set', help='data directory')
    parser.add_option('--sample-dir', dest='sample_dir', default='./samples', help='sample directory')
    parser.add_option('--cpt-dir', dest='cpt_dir', default='./checkpoints', help='checkpoint directory')
    parser.add_option('--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('--test', action='store_false', default=True, help='testing mode')
    parser.add_option('--pth', default='CP10.pth', help='pth')

    (options, args) = parser.parse_args()
    return options

def showSample(img, mask, out, epoch, sample_dir, train):
    t = 'train'
    if not train:
        t = 'test'

    img = img.cpu().detach().numpy()
    img_input = np.copy(img)
    mask = np.tile(mask, 3)
    img_input[mask<1] = 0

    plt.subplot(1, 3, 1).set_title(str(epoch)+'_'+t+'_gt')
    plt.imshow(img)
    plt.subplot(1, 3, 2).set_title(str(epoch)+'_'+t+'_in')
    plt.imshow(img_input)
    plt.subplot(1, 3, 3).set_title(str(epoch)+'_'+t+'_out')
    plt.imshow(out.cpu().detach().numpy())

    plt.savefig(os.path.join(sample_dir,str(epoch)+'_'+t+'.png'))

if __name__ == '__main__':
    args = getArgs()

    net = UNet()

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    if not os.path.exists(args.cpt_dir):
        os.makedirs(args.cpt_dir)

    trainNet(net=net,
        data_dir=args.data_dir,
        sample_dir=args.sample_dir,
        cpt_dir=args.cpt_dir,
        epochs=args.epochs,
        gpu=args.gpu,
        train=args.test,
        pth=args.pth)