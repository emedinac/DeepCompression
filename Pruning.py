import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from termcolor import colored

from deep_utils import *

# this code implements pruning using register buffer feature to save input mask
class PruneNetwork(nn.Module):
    def __init__(self, model):
        super(PruneNetwork, self).__init__()
        # get the model ready to be pruned.
        self.base_model = model
        self.all_layers = [n for n,i in self.base_model.named_parameters()]

        self.modules = []
        for i in range(len(self.all_layers)):
            self.modules.append(self.all_layers[i].split('.')[:-1])

    def compute_mask(self, layer, weights):
        thresh = weights.std()
        m1 = weights.abs() < thresh
        mask = torch.ones(weights.size()).cuda()
        mask = mask-m1.float()
        print('Layer:\t',self.all_layers[layer], '\tThreshold:\t', thresh.item(),'\tPruned weights (%):\t', colored((1-mask.mean().item())*100,'red') )
        return mask

    def Prune(self):
        # compute the mask for the weights
        for layer in range(len(self.all_layers)):
            # print('I am here: ',self.all_layers[layer])
            levels = self.modules[layer]
            # iterative layer reading.
            current_layer = self.base_model

            for k in range(len(levels)):
                current_layer = current_layer._modules.get(levels[k])

            if 'weight' in self.all_layers[layer]:
                weights = current_layer.weight.data
                # print('weights mean:  ', weights.mean())
                # bias = current_layer.bias.data
                # compute the mask
                mask = self.compute_mask(layer, weights)
                # print('prunned weights (%):  ', colored((1-mask.mean())*100,'red'))
                # print(weights.std().item(), mask.sum().item()  , current_layer.weight.abs().sum().item())
                # mask the weights
                current_layer.weight.data =  weights*mask
                # print(self.base_model.state_dict()[self.all_layers[layer]].abs().sum().item())
                
                # print(((current_layer.weight.data==0)).cpu().float().mean()*100)
                # print(current_layer)
                # print('        \n')
    def forward(self, x):
        return self.base_model(x)

# Simple test playing on Pytorch
# net.cpu()
# parameters = net.state_dict()
# layers = list(filter(lambda x:'features' in x or 'classifier' in x, parameters.keys()))
# for i in range(len(layers)):
#     print(layers[i], parameters[layers[i]].numpy().max(),'            ' ,(parameters[layers[i]].numpy()==np.inf).mean()*100)
# net.cuda()