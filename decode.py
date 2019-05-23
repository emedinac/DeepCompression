'''
If you find Deep Compression useful in your research, please consider citing the paper:

@inproceedings{han2015learning,
  title={Learning both Weights and Connections for Efficient Neural Network},
  author={Han, Song and Pool, Jeff and Tran, John and Dally, William},
  booktitle={Advances in Neural Information Processing Systems (NIPS)},
  pages={1135--1143},
  year={2015}
}


@article{han2015deep_compression,
  title={Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding},
  author={Han, Song and Mao, Huizi and Dally, William J},
  journal={International Conference on Learning Representations (ICLR)},
  year={2016}
}

A hardware accelerator working directly on the deep compressed model:

@article{han2016eie,
  title={EIE: Efficient Inference Engine on Compressed Deep Neural Network},
  author={Han, Song and Liu, Xingyu and Mao, Huizi and Pu, Jing and Pedram, Ardavan and Horowitz, Mark A and Dally, William J},
  journal={International Conference on Computer Architecture (ISCA)},
  year={2016}
}



'''

"""
VGG (
  (features): Sequential (
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))               -----every convlayer has bias and weights: 64x3x3x3
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)                          -----every batchnorm has bias and weights: 64x2
    (2): ReLU (inplace)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
    (5): ReLU (inplace)
    (6): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (9): ReLU (inplace)
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (12): ReLU (inplace)
    (13): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (16): ReLU (inplace)
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (19): ReLU (inplace)
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (22): ReLU (inplace)
    (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (25): ReLU (inplace)
    (26): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (29): ReLU (inplace)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (32): ReLU (inplace)
    (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (35): ReLU (inplace)
    (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (38): ReLU (inplace)
    (39): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (42): ReLU (inplace)
    (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (45): ReLU (inplace)
    (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (48): ReLU (inplace)
    (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (51): ReLU (inplace)
    (52): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (classifier1): Linear (512 -> 4096)
  (classifier2): Linear (4096 -> 4096)
  (classifier3): Linear (4096 -> 10)
  (dropout): Dropout (p = 0.2, inplace)
)

"""
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
def debug_good_model(net):
    from utils import progress_bar
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def test(epoch):
        net.eval()
        criterion = nn.CrossEntropyLoss()
        net.cuda()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.module if use_cuda else net,
                'acc': acc,
                'epoch': epoch,
            }
        return test_loss/(batch_idx+1), 100.*correct/total
    return test(0)






import sys
import os
import numpy as np
import _pickle as pickle# to serialize objects
import gzip # to decompress
import torch
# import torchvision
sys.path.append('../S3Pool/')
from models import *

path = '../S3Pool/checkpoint/VGG19_BN_10.t7'
checkpoint = torch.load(path)
net = checkpoint['net'].cpu()
best_acc = checkpoint['acc']

parameters_sizes=[]
for i in net.parameters():
    parameters_sizes.append(i.data.shape)


def binary_to_net(weights, spm_stream, ind_stream, codebook, num_nz):
    bits = np.log2(codebook.size)
    if bits == 4:
        slots = 2
    elif bits == 8:
        slots = 1
    else:
        print("Not impemented,", bits)
        sys.exit()
    code = np.zeros(weights.size, np.uint8) 

    # Recover from binary stream
    spm = np.zeros(num_nz, np.uint8)
    ind = np.zeros(num_nz, np.uint8)
    if slots == 2:
        spm[np.arange(0, num_nz, 2)] = spm_stream % (2**4)
        spm[np.arange(1, num_nz, 2)] = spm_stream / (2**4)
    else:
        spm = spm_stream
    ind[np.arange(0, num_nz, 2)] = ind_stream% (2**4)
    ind[np.arange(1, num_nz, 2)] = ind_stream/ (2**4)


    # Recover the matrix
    ind = np.cumsum(ind+1)-1
    code[ind] = spm
    data = np.reshape(codebook[code], weights.shape)
    np.copyto(weights, data)




test_loss , test_acc  = debug_good_model(net)
print(test_loss , test_acc )
print('===========')
net.cpu()

parameters = net.state_dict()
layers = list(filter(lambda x:'features' in x or 'classifier' in x, parameters.keys()))
fin = open(path, 'rb')
nz_num = np.fromfile(fin, dtype = np.uint32, count = len(layers))

for idx, layer in enumerate(layers):
    # print("Reconstruct layer", layer)
    # print("Total Non-zero number:", nz_num[idx])
    if 'classifier' in layer:
        bits = 4
    else:
        bits = 8

    if ('bias' or 'running_') in layer:
        bias = np.array(parameters[layer].size())[0] # do nothing for bias terms...
        np.copyto(parameters[layer].numpy(), bias)
    else:
        codebook_size = 2 ** bits
        codebook = np.fromfile(fin, dtype = np.float32, count = codebook_size)

        spm_stream = np.fromfile(fin, dtype = np.uint8, count = int((nz_num[idx]-1) / (8/bits) + 1))
        ind_stream = np.fromfile(fin, dtype = np.uint8, count = int((nz_num[idx]-1) / 2+1))

        binary_to_net(parameters[layer].numpy(), spm_stream, ind_stream, codebook, nz_num[idx])
        print('entro')
# net.save(target)
# print("All done! See your output caffemodel and test its accuracy.")


