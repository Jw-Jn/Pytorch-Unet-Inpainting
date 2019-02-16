import torch
import numpy as np
from optparse import OptionParser
import os
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import UNet
import dataset
import random

def trainNet(net, data_dir, epochs=100, gpu=True, train=True, pth_dir=None):

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

                out = net.forward(img_input)

                loss = criterion(out, torch.transpose(img, 1, 3))

                epoch_loss += loss.item()
                
                loss.backward()
                optimizer.step()

                print('Training sample %d / %d - Loss: %.6f' % (i+1, 100, loss.item()))
                
            print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))

            idx = random.randint(0, 15) # batch size -1
            showSample(img[idx], mask[idx], torch.transpose(out, 1, 3)[idx], (epoch+1), train=True)

            if (epoch+1)%10 == 0:
                torch.save(net.state_dict(), os.path.join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
                print('Checkpoint %d saved !' % (epoch + 1))

    else:
        test_dataset = dataset.InpaintingDataSet(os.path.join(data_dir, 'test.png'), 10)
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)
        print('test items:', len(test_dataset))

        print('Testing')
        net.load_state_dict(torch.load(pth_dir))
        net.eval()

        with torch.no_grad():
            for i, (img, mask) in enumerate(test_data_loader):

                img_input = torch.cat([img, mask], dim=-1)
                img_input = torch.transpose(img_input, 1, 3)

                if gpu:
                    img_input = Variable(img_input.cuda())

                out = net.forward(img_input)

                showSample(img[0], mask[0], torch.transpose(out, 1, 3)[0], i, train=False)


def getArgs():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='./inpainting_set', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('--test', action='store_false', default=True, help='testing mode')
    parser.add_option('--pth', default='./data/cells/checkpoints/CP60.pth', help='pth directory')

    # parser.add_option('--frz', action='store_true', default=False, help='flip rotate zoom')
    # parser.add_option('--gamma', action='store_true', default=False, help='gamma correction')
    # parser.add_option('--deform', action='store_true', default=False, help='deformation')

    (options, args) = parser.parse_args()
    return options

def showSample(img, mask, out, epoch, train):
    img_input = np.copy(img)
    mask = np.tile(mask, 3)
    img_input[mask<1] = 0

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(img_input)
    plt.subplot(1, 3, 3)
    plt.imshow(out.cpu().detach().numpy())
    if train:
        plt.savefig('./samples/train'+str(epoch)+'.png')
    else:
        plt.savefig('./samples/test'+str(epoch)+'.png')


    # fig = plt.figure()
    # ax1 = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)
    # ax1.title.set_text(str(epoch)+'_'+'train'+'_'+'gt')
    # ax2.title.set_text(str(epoch)+'_'+'train'+'_'+'gt')
    # ax3.title.set_text(str(epoch)+'_'+'train'+'_'+'gt')
    # # plt.show()
    # plt.savefig('./samples/train'+str(epoch)+'.png')


if __name__ == '__main__':
    args = getArgs()

    net = UNet()

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    trainNet(net=net,
        data_dir=args.data_dir,
        epochs=args.epochs,
        gpu=args.gpu,
        train=args.test,
        pth_dir=args.pth)
        # flip=args.frz,
        # rotate=args.frz,
        # zoom=args.frz,
        # gamma_c=args.gamma,
        # deform=args.deform)


## visualize some data
# _, (img, mask) = next(enumerate(train_data_loader))
# nd_img = img.cpu().numpy()
# nd_mask = mask.cpu().numpy()
# showSample(nd_img, nd_mask)