import os
import time
import _pickle as pickle# to serialize objects
import gzip # to decompress
import argparse
import numpy as np
from termcolor import colored

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

from Pruning import PruneNetwork
from Sharing import SharingWeightsNetwork
from deep_utils import * # CudaPytorchunique, CountAllWeights, CountZeroWeights, SumAllWeights
from models import *
from learning import *
import pdb 

# lr1 = '0.05,0.025,0.025,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.0025,0.0025,0.001,0.0001,0.0001'
# lr2 = '0.00001,0.00001,0.0001,0.0001, 0.005,0.005,0.005 , 0.001,0.001,0.001,0.001'
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr1', default='0.02,0.02,0.015,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.0025,0.0025,0.001,0.0001,0.0001', type=str, help='learning rate for Stage 1')
parser.add_argument('--lr2', default='0.00001,0.00001,0.0001,0.0001, 0.005,0.005,0.005 , 0.001,0.001,0.001,0.001', type=str, help='learning rate for Stage 2')
parser.add_argument('--lrsch_1', default='4,11,17,22', type=str, help='learning rate schedule for Stage 1')
parser.add_argument('--lrsch_2', default='8,13,18,23', type=str, help='learning rate schedule for Stage 2')
parser.add_argument('--lcluster', default='4,5,7,9,  11,13,15,  18,20,25,35', type=str, help='cluster list for Stage 2')
parser.add_argument('--cluster_method', default='GMM', type=str, help='clustering method used for Stage 2 - available methods are: kmeans, GMM, GMM-kmeans')
parser.add_argument('--factor', default=0.2, type=float, help='Scale factor for optimizer scheduler')
parser.add_argument('--epoch', default=25, type=int, help='number of epochs per pruning iteration')
parser.add_argument('--iter', default=25, type=int, help='number of iterations for pruning')
parser.add_argument('--apply_pruning', default=False, type=bool, help='apply pruning stage')
parser.add_argument('--apply_sharing', default=True, type=bool, help='apply sharing stage')
parser.add_argument('--gpus', default=None, type=int, help='Use more gpus?')

# clusters_list = [4,5,7,9,  11,13,15,  18,20,25,35]
# list_lr = [0.00001, 0.00001, 0.0001, 0.0001,    0.00005, 0.00005, 0.00005,    0.0001, 0.0001, 0.0001, 0.0001]
# clusters_list = [35]
# list_lr = [0.00005]
start_iter = 0

args = parser.parse_args()
iterations = args.iter
Apply_Pruning = args.apply_pruning
Apply_Sharing = args.apply_sharing
cluster_method = args.cluster_method
list_lr1 = [float(i) for i in args.lr1.split(',')]
list_lr2 = [float(i) for i in args.lr2.split(',')]
schedule1 = [int(i) for i in args.lrsch_1.split(',')]
schedule2 = [int(i) for i in args.lrsch_2.split(',')]
clusters_list = [int(i) for i in args.lcluster.split(',')]
factor = args.factor
# torch.nn.Module.dump_patches = True
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print("learning rate:", list_lr1[0])
if len(list_lr1) < iterations:
    raise Exception('lenght of the learning rate list must be equal to iteration numbers')
###############################################
################################################ Data and model name
############################################### 
"""
A logger strategy is strongly needed =( . The future experiments have the research logger available in:
https://github.com/emedinac/Research-purpose_Logger
"""
original_namemodel='VGG19_base_10'
description =  ['VGG for CIFAR10 database \n',
                'optim.SGD(net.parameters(), lr=lr, momentum=0.9,) \n',
                'run main.py \n',
                'no L2, using data augmentation ',
                ]

# namemodel = 'VGG16'
# description = 'Model employed for Deep Compression'
###############################################
###############################################
###############################################

def DeepCompression_FirstStage(net):
    global best_acc, namemodel
    accuracies = []
    total_weights, zero_weights = [], []

    
    # Initial weights and accuracy of the model.
    test_loss , test_acc  = env.test(0)
    accuracies.append(best_acc)
    print('Current Network Accuracy: ', colored(test_acc,'blue'))
    print(colored('Before Pruning','green'))
    all_parameters,b,c = CountZeroWeights(net)
    SumAllWeights(net)
    total_weights.append(b)
    zero_weights.append(c)

    tr_loss = []; tr_acc = []; te_loss = []; te_acc = [];  
    for it in range(start_iter+1,iterations):
        namemodel = original_namemodel+'_Pruned'+str(it).zfill(2)
        print('=========================')
        print("This will be the next file: ", namemodel)
        print('=========================')
        print(colored('Iteration: ','green'), colored(it,'green'))
        lr = list_lr1[it]
        env.lr(lr)

        print(colored('Pruning network',"yellow"))
        net = PruneNetwork(net) # Pruning magic is performed inside 8-)
        net.Prune()
        env.InsertNet(net)
        print(colored('After Pruning','green'))
        a,b,c = CountZeroWeights(env.net)
        SumAllWeights(env.net)
        test_loss , test_acc  = env.test(0)
        print('Accuracy: ', colored(test_acc,'red'))

        total_weights.append(b)
        zero_weights.append(c) 
        env.best_acc = 0
        env.printlr()
        for epoch in range(start_epoch, start_epoch+args.epoch):
            if epoch in schedule1:
                env.scheduler()
            train_loss, train_acc = env.train(epoch,'Pruning')
            test_loss , test_acc  = env.test(epoch, namemodel)

            tr_loss.append(train_loss); tr_acc.append(train_acc); te_loss.append(test_loss); te_acc.append(test_acc)

        accuracies.append(env.best_acc) # save accuracy list
        net = Load_model(namemodel)
        CountZeroWeights(net)
        SumAllWeights(net)

        train_loss = np.array(tr_loss)[:,None]
        train_acc = np.array(tr_acc)[:,None]
        test_loss = np.array(te_loss)[:,None]
        test_acc = np.array(te_acc)[:,None]

    
    net = Load_model(namemodel)
    print(colored('After Training','green'))
    test_loss , test_acc  = env.test(epoch, namemodel)
    CountZeroWeights(net)
    SumAllWeights(net)

    return net, (train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, zero_weights)


def DeepCompression_SecondStage(net):
    global namemodel
    # Initial weights and accuracy of the model.
    accuracies = []
    total_non_weights, unique_weights = [], []
    accuracies.append(env.best_acc)

    all_parameters,b,c = CountAllWeights(net)
    total_non_weights.append(b)
    unique_weights.append(c)

    # schedule = [8,18]
    tr_loss = []; tr_acc = []; te_loss = []; te_acc = [];  
    for lr_i, k in enumerate(clusters_list):
        namemodel = original_namemodel+'_Shared'+str(k).zfill(2)
        print('=========================')
        print('=========================')
        print(colored('Cluster: ','green'), colored(k,'green'))
        lr = list_lr2[lr_i]
        env.lr(lr)
        
        print('Sharing weights in network')
        net = SharingWeightsNetwork(net,cluster=k)  # Sharing magic is performed inside 8-)
        net.Sharing(cluster_method) # Method 1
        env.InsertNet(net)
        print(colored('After Sharing weights','green'))
        a,b,c = CountAllWeights(env.net)
        test_loss , test_acc  = env.test(net, 0)
        print('Accuracy: ', colored(test_acc,'red'))
        total_non_weights.append(b)
        unique_weights.append(c) 
        env.best_acc = 0

        for epoch in range(start_epoch, start_epoch+args.epoch):
            if epoch in schedule2:
                env.scheduler()
            train_loss, train_acc = env.train(epoch,'WeightSharing')
            test_loss , test_acc  = env.test(epoch, namemodel)

            tr_loss.append(train_loss); tr_acc.append(train_acc); te_loss.append(test_loss); te_acc.append(test_acc)

        accuracies.append(env.best_acc) # save accuracy list
        namemodel =  original_namemodel+'_Pruned'+str(iterations-1).zfill(2) # Last prunned network
        net = Load_model(namemodel)
        CountAllWeights(net)
        SumAllWeights(net)

    train_loss = np.array(tr_loss)[:,None]
    train_acc = np.array(tr_acc)[:,None]
    test_loss = np.array(te_loss)[:,None]
    test_acc = np.array(te_acc)[:,None]
    return net, (train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_non_weights, unique_weights)



# Load checkpoint.
def Load_model(namemodel):
    global net_acc
    assert os.path.isdir('./checkpoint/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+namemodel+'.t7')
    # net = models.CustomVGG('6ls3',[16,8,4,2,1])
    net = VGG('VGG19',10).cuda()
    print(colored('==> Resuming from checkpoint..'+str(namemodel),"green"))
    net.load_state_dict(checkpoint['net']) # net is the state_dict
    net_acc = checkpoint['acc']
    if args.gpus: net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    return net

#################################################
################################################# MAIN 
#################################################
if __name__ == "__main__":
    # Load Model
    namemodel = original_namemodel
    if start_iter: net = Load_model(namemodel+'_Pruned'+str(start_iter).zfill(2)) 
    else: net = Load_model(namemodel) 

    # Setting ENV #
    DATA = 'CIFAR10'
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9,) # LR can be set with any value. It is adapted in each network training
    env = LearningProcess(DATA, nn.CrossEntropyLoss(), optimizer, gpus=args.gpus)
    env.schedule_factor = factor

    # For some models, not important, but useful to understand why Batch normalization goes to INF in the Running_Var (ref: BN paper)
    # no_inf = net._modules.get('module')._modules.get('features')._modules.get('8')
    # no_inf.running_var[no_inf.running_var==np.inf] = 1e5 # for INF problems during training. (in case model overfits)
    print(colored('Original Model','green'))
    env.InsertNet(net)
    test_loss , test_acc  = env.test(0)
    CountZeroWeights(env.net)

    if Apply_Pruning: # Apply or load...
        print(colored('\n\nFisrt Stage: \n\n','green'))
        net, Variables_to_save = DeepCompression_FirstStage(net)
        train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, zero_weights = Variables_to_save
        pickle.dump(Variables_to_save, gzip.open('./checkpoint/'+namemodel+'.p.gz', 'wb'))
    else:
        print(colored('\n\nLoading Pruning Stage: \n\n','green'))
        Variables_to_save = pickle.load(gzip.open('./checkpoint/'+namemodel+'_Pruned'+str(iterations-1).zfill(2)+'.p.gz', 'rb'))
        train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, zero_weights = Variables_to_save

    namemodel = original_namemodel+'_Pruned'+str(iterations-1).zfill(2)
    net = Load_model(namemodel)
    print(colored('After Pruning','green'))
    test_loss , test_acc  = env.test(net, 0)
    print(test_loss , test_acc)
    CountZeroWeights(net)

    pdb.set_trace() ########################################################################################################


    if Apply_Sharing: # Apply or load...
        print(colored('\n\nSecond Stage: \n\n','green'))
        net, Variables_to_save = DeepCompression_SecondStage(net)
        train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_non_weights, unique_weights = Variables_to_save
        pickle.dump(Variables_to_save, gzip.open('./checkpoint/Sharing_'+namemodel+'.p.gz', 'wb'))
    else:
        print(colored('\n\nLoading Sharing Stage: \n\n','green'))
        Variables_to_save = pickle.load(gzip.open('./checkpoint/Sharing_'+namemodel+'.p.gz', 'rb'))
        train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_non_weights, unique_weights = Variables_to_save

    namemodel = original_namemodel+'_Shared'+str(clusters_list[-1]).zfill(2)
    net = Load_model(namemodel)
    env.InsertNet(net)
    print(colored('After Training','green'))
    test_loss , test_acc  = env.test(0)
    CountAllWeights(net)




