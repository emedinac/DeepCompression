import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from termcolor import colored

# Not efficient but works well... For counting.
weights_list = 0
def weights(net):
    global weights_list
    parameters = net.state_dict()
    layers = list(parameters.keys())
    weights_list = []
    for i in range(len(layers)):
        temp = parameters[layers[i]].cpu().numpy()
        weights_list.append(temp)
    return weights_list

def CudaPytorchunique(w):
    w,_ = w.view(-1).sort()
    unique_w = w[:1]
    w = w[unique_w!=w]
    while(len(w)):
        unique_w = torch.cat((unique_w,w[:1]),dim=0)
        w = w[w[:1]!=w]
    return unique_w


def CountAllWeights(net):
    parameters = net.state_dict()
    layers = list(net.state_dict().keys())
    all_parameters, total_non_weights, unique_weights = 0, 0, 0
    for i in range(len(layers[:-1])):
        param = parameters[layers[i]].cpu().numpy()
        all_parameters += np.size(param)

        if 'weight' in layers[i]:
            non_zero = param[param!=0]
            total_non_weights += np.size(non_zero)
            unique_weights += np.size(np.unique(non_zero))
    print('total parameters: {},\nTotal non-weights: {},\nUnique weights: {},\nUnique non-zero weights rate: {},\nUnique non-zero parameters rate in model: {}'.format(
        all_parameters, total_non_weights, unique_weights, colored(unique_weights/total_non_weights*100, 'blue'), colored(unique_weights/all_parameters*100, 'green')))
    return all_parameters, total_non_weights, unique_weights

def CountZeroWeights(net):
    parameters = net.state_dict()
    layers = list(parameters.keys())
    all_parameters = total_weights = zero_weights= 0 
    for i in range(len(layers)):
        # print(layers[i], parameters[layers[i]].numpy().max(),'            ' ,(parameters[layers[i]].numpy()==0).mean()*100)
        all_parameters += np.size(parameters[layers[i]].cpu().numpy())
        # print(layers[i],'\t\t', all_parameters, parameters[layers[i]].cpu().numpy().shape)
        if 'weight' in layers[i]:
            temp = parameters[layers[i]].cpu().numpy()
            total_weights += np.size(temp)
            zero_weights += (temp==0).sum()
    print('total parameters: {},\nTotal weights: {},\nZero weights: {},\nZero weights rate: {},\nZero weights rate in model: {}'.format(
        all_parameters, total_weights, zero_weights, colored(zero_weights/total_weights*100, 'blue'), colored(zero_weights/all_parameters*100, 'green')))
    return all_parameters, total_weights, zero_weights

def SumAllWeights(net):
    parameters = net.state_dict()
    layers = list(parameters.keys())
    all_parameters = 0 
    for i in range(len(layers)):
        if 'weight' in layers[i] or 'bias' in layers[i]:
            all_parameters += parameters[layers[i]].abs().sum().item()
    print(colored('sum (absolute) values from all parameters: {}'.format(all_parameters),"yellow"))
    return all_parameters
