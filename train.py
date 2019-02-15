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

def train_net(net, data_dir, epochs=5, gpu=True, train=True, pth_dir='./data/cells/checkpoints/CP10.pth'):
            #   flip=False,
            #   rotate=False,
            #   zoom=False,
            #   gamma_c=False,
            #   deform=False):

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    max_epochs = 80
    learning_rate = 0.0001
    
    # transform = ['flip', 'crop']

    ## visualize some data
    # _, (img, mask) = next(enumerate(train_data_loader))
    # nd_img = img.cpu().numpy()
    # nd_mask = mask.cpu().numpy()
    # dp.show_landmarks(nd_img, nd_lm)

    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

    if train:
        train_dataset = dataset.InpaintingDataSet(os.path.join(data_dir, 'train.png'))
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=16,
                                                        shuffle=False,
                                                        num_workers=0)
        print('train items:', len(train_dataset))

        for epoch in range(0, max_epochs):
            print('Epoch %d/%d' % (epoch + 1, epochs))
            print('Training...')
            net.train()

            epoch_loss = 0

            for i, (img, mask) in enumerate(train_data_loader):
            
                optimizer.zero_grad()

                img = torch.transpose(img, 1, 3)

                if gpu:
                    img = Variable(img.cuda())

                out = net.forward(img)

                loss = criterion(out.view((-1, 2, 7)), img)

                epoch_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                if (epoch+1)%10 == 0:
                    torch.save(net.state_dict(), os.path.join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
                    print('Checkpoint %d saved !' % (epoch + 1))
                print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))

        # plt.show()
        # plt.savefig(file_name+'.jpg')

    else:
        test_dataset = dataset.InpaintingDataSet(os.path.join(data_dir, 'test.png'))
        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)
        print('train items:', len(train_dataset))

        print('Testing')
        net.load_state_dict(torch.load(pth_dir))
        net.eval()

        with torch.no_grad():
            for i, (img, mask) in enumerate(test_data_loader):
                shape = img.shape
                img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
                if gpu:
                    img_torch = img_torch.cuda()

                pred = net(img_torch)

                # plt.subplot(1, 3, 1)
                # plt.imshow(img*255.)
                # plt.subplot(1, 3, 2)
                # plt.imshow((label-1)*255.)
                # plt.subplot(1, 3, 3)

                # plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
                # # plt.show()
                # plt.savefig('test_'+str(i)+'.png')


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='./inpainting_set', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('--test', action='store_false', default=True, help='testing mode')
    parser.add_option('--pth', default='./data/cells/checkpoints/CP60.pth', help='pth directory')

    # parser.add_option('--frz', action='store_true', default=False, help='flip rotate zoom')
    # parser.add_option('--gamma', action='store_true', default=False, help='gamma correction')
    # parser.add_option('--deform', action='store_true', default=False, help='deformation')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet()

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
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