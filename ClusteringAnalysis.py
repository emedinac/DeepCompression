'''
This code asumes Pruning stage was performed. Analysis of several clustering techniques are implemented to test what is the best
solution for Weight Sharing method (as Deep Compression mentioned). The clustering techniques employed are the following:
- KMeans
- AffinityPropagation
- MeanShift
- SpectralClustering
- Ward
- AgglomerativeClustering
- DBSCAN
- Birch
- GaussianMixture
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

import argparse
import numpy as np

import _pickle as pickle# to serialize objects
import gzip # to decompress
import os, sys
from termcolor import colored, cprint

from utils import progress_bar, ReadDataBase
from Pruning import PruneNetwork, CountZeroWeights
sys.path.append('../S3Pool/')
from models import *

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default='0.1,0.01', type=str, help='learning rate')
parser.add_argument('--lrsch', default='10,20', type=str, help='learning rate schedule')
parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
parser.add_argument('--iter', default=2, type=int, help='number of iterations for pruning')

args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

iterations = args.iter

list_lr = [float(i) for i in args.lr.split(',')]
schedule = [int(i) for i in args.lrsch.split(',')]
print("learning rate:", list_lr[0])
if len(list_lr) != iterations:
    raise Exception('lenght of the learning rate list must be equal to iteration numbers')
###############################################
################################################ Data and model name
###############################################
namemodel='VGG19_BN_10_Pruned24'
description =  ['VGG for CIFAR10 database \n',
                'optim.SGD(net.parameters(), lr=lr, momentum=0.9) \n',
                'run main.py --lr=0.05,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.001,0.001 --lrsch=3,10,16 --epoch=20 --iter=20 \n',
                'no L2, no dropout in linear, using batchnorm, using GAP, using data augmentation ']

# namemodel = 'VGG16'
# description = 'Model employed for Deep Compression'
###############################################
###############################################
###############################################
DATA = 'CIFAR10'
trainloader, testloader = ReadDataBase(DATA)




# Load checkpoint.
def Load_model(namemodel):
    global net_acc
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./checkpoint/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+namemodel+'.t7')
    net = checkpoint['net']
    net_acc = checkpoint['acc']
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return net

def Pruning():
    global net
    print('Pruning network')
    net = PruneNetwork(net)
    net.Prune()
    print(colored('After Pruning','green'))
    test_loss , test_acc  = test(0)
    a,b,c = CountZeroWeights()
    print('Accuracy: ', colored(test_acc,'blue'))
    return a,b,c

# Load Model
net = Load_model(namemodel)
criterion = nn.CrossEntropyLoss()

accuracies = []
all_parameters, total_weights, zero_weights = [], [], []

# Initial weights and accuracy of the model.
a,b,c = CountZeroWeights(net)

def ReadAllWeights(net):
    parameters = net.state_dict()
    all_layers = list(net.state_dict().keys())
    modules = []
    for i in range(len(all_layers)):
        modules.append(all_layers[i].split('.')[:-1])

    weights_grouped_by_layers = []
    layer_name = []
    module_name = []
    for layer in range(len(all_layers)):
        levels = modules[layer]
        current_layer = net
        for k in range(len(levels)):
            current_layer = current_layer._modules.get(levels[k])
        if 'weight' in all_layers[layer]:
            weights = current_layer.weight.data.cpu().numpy().flatten()
            print(current_layer, ':  \t\t\t',len(weights))
            weights_grouped_by_layers.append(  weights   )
            layer_name.append(all_layers[layer])
            module_name.append(current_layer)
    return module_name, layer_name, weights_grouped_by_layers

module_name, layer_name, weights = ReadAllWeights(net)


font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 15}
plt.rc('font', **font)
axis = 10
axisplot = 200
for i in range(len(weights)):
    fig, ax = plt.subplots(figsize=(15,10), dpi=350)
    plt.title('Weights Density Distribution \n'+str(layer_name[i])+'\n'+str(module_name[i]))
    ax.set_xlabel('Weight values', color='k')
    ax.set_ylabel('Density', color='b')

    minX, maxX = np.floor(weights[i].min()), np.ceil(weights[i].max())
    bins_plot = np.linspace(minX, maxX, axisplot)
    hist, bins = np.histogram(weights[i], bins=bins_plot)

    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2
    hist[np.abs(center).min()==np.abs(center)]=0
    ax.bar(center, hist, align='center', width=width)

    
    ax.set_xticks(np.arange(minX, maxX , (maxX-minX)/axis) )
    # plt.show()
    plt.savefig('./Distributions/Weight_'+str(i).zfill(2)+'.png', bbox_inches='tight', dpi=300)
    plt.close()


del net, module_name, layer_name, trainloader, testloader
del torch, nn, Variable, transforms, optim, cudnn, progress_bar, ReadDataBase, PruneNetwork, CountZeroWeights, VGG


import time
import warnings

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

# ============
# Set up cluster parameters
# ============
axis = 10
axisplot = 200
num_clusters = [2,3,4,5,6,10,13,15]
for layer in [0,2,4,5,6,7,10,13,14,15,20,31,33]:
    print('==============layer :', layer)
    plt.figure(figsize=(60, 8*len(num_clusters)), dpi=350)
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    datasets = [ (weights[layer], {})]

    plot_num = 1
    for num, k in enumerate(num_clusters):
        print(num,' :  ', k, ' clusters')
        default_base = {'quantile': 0.4,
                        'eps': .1,
                        'damping': .9,
                        'preference': -200,
                        'n_neighbors': 20,
                        'n_clusters': k}

        for index, (dataset, algo_params) in enumerate(datasets):
            
            # update parameters with dataset-specific values
            params = default_base.copy()
            params.update(algo_params)

            X = dataset[dataset!=0][:,None]

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph( X, n_neighbors=params['n_neighbors'], include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # ============
            # Create cluster objects
            # ============
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.KMeans(n_clusters=params['n_clusters'])
            ward = cluster.AgglomerativeClustering( n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
            spectral = cluster.SpectralClustering( n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=params['eps'])
            affinity_propagation = cluster.AffinityPropagation( damping=params['damping'], preference=params['preference'])
            average_linkage = cluster.AgglomerativeClustering( linkage="average", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)
            birch = cluster.Birch(n_clusters=params['n_clusters'])
            gmm = mixture.GaussianMixture( n_components=params['n_clusters'], covariance_type='full')

            clustering_algorithms = (
                ('KMeans', two_means),
                ('AffinityPropagation', affinity_propagation),
                ('MeanShift', ms),
                ('SpectralClustering', spectral),
                ('Ward', ward),
                ('AgglomerativeClustering', average_linkage),
                ('DBSCAN', dbscan),
                ('Birch', birch),
                ('GaussianMixture', gmm)
            )

            for name, algorithm in clustering_algorithms:
                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning)
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding" +
                        " may not work as expected.",
                        category=UserWarning)
                    algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)

                plt.subplot(len(datasets)*len(num_clusters), len(clustering_algorithms), plot_num)

                if index == 0:
                    plt.title(name, size=18)
                    plt.ylabel(str(k)+' clusters', color='k', size=18)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))

                
                # plt.scatter(X, np.zeros_like(X) + 0., s=10, color=colors[y_pred])

                minX, maxX = np.floor(X.min()), np.ceil(X.max())
                bins_plot = np.linspace(minX, maxX, axisplot)
                hist, bins = np.histogram(X, bins=bins_plot)
                minY, maxY = hist.min(), hist.max()

                width = np.diff(bins)
                center = (bins[:-1] + bins[1:]) / 2
                hist[np.abs(center).min()==np.abs(center)]=0
                color = []
                print(name)
                for p in range(len(hist)):
                    c = y_pred[np.logical_and(X>bins[p] , X<bins[p+1])[:,0]]
                    if np.size(c)==0:
                        color.append(0)
                        # print('-')
                    else:
                        color.append( int(np.round(c.mean())) )
                        # print(color[-1])
                # clusters = algorithm.cluster_centers_
                # length = len(clusters)
                plt.bar(center, hist, align='center', width=width, color=colors[color])
                # plt.stem(clusters, [hist.min()]*len(clusters), markerfmt=' ', linefmt='ko-')
                # plt.bar(clusters, [hist.max()]*length, align='center', width=[width[0]/2]*length, color='k'*length) # color=colors[np.unique(y_pred)]
                # plt.plot(clusters, [-1]*len(clusters), 'ko', markersize=10)
                plt.xticks(np.arange(minX, maxX , (maxX-minX)/axis) )
                # plt.yticks(np.arange(minY, maxY+10 , 15 ))
                # plt.ylim(minY-5, maxY+10)
                ## plt.xlim(-2.5, 2.5)

                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                plt.text(.99, .01, str(np.unique(y_pred)),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='left')
                plot_num += 1

    plt.savefig('./Distributions/Clustered'+str(layer).zfill(2)+'.png', bbox_inches='tight', dpi=300)
    plt.close()
    plt.close()
    plt.close()
# plt.show()