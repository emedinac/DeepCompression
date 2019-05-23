import matplotlib.pyplot as plt
import _pickle as pickle# to serialize objects
import gzip # to decompress
import os, sys
import numpy as np
from termcolor import colored, cprint

import torch
import torch.backends.cudnn as cudnn
sys.path.append('../S3Pool/')
from models import *
from Pruning import PruneNetwork, CountZeroWeights
from Sharing import SharingWeightsNetwork, CountAllWeights

# Load checkpoint.
def Load_model(namemodel):
    global net_acc
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./checkpoint/'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./'+namemodel+'.t7')
    net = checkpoint['net']
    net_acc = checkpoint['acc']
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    return net

original_namemodel='checkpoint/Pruned_woL2/VGG19_BN_10'
namemodel = original_namemodel+'_Pruned24'
train_iter_data = pickle.load(gzip.open('./'+namemodel+'.p.gz', 'rb'))
train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, zero_weights = train_iter_data




def compute_mask(layer, weights):
    thresh = weights.std()
    m1 = weights.abs() < thresh
    mask = torch.ones(weights.size()).cuda()
    mask = mask-m1.float()
    print('Layer:\t',layer, '\tThreshold:\t', thresh,'\t\tPrunned weights (%):\t', colored((1-mask.mean())*100,'red') )
    return mask

def CountZeroWeights(net):
    parameters = net.state_dict()
    layers = list(filter(lambda x:'features' in x or 'classifier' in x, parameters.keys()))
    all_parameters, total_weights, zero_weights= 0, 0, 0
    for i in range(len(layers)):
        # print(layers[i], parameters[layers[i]].numpy().max(),'            ' ,(parameters[layers[i]].numpy()==0).mean()*100)
        all_parameters += np.size(parameters[layers[i]].cpu().numpy())
        # print(layers[i],'\t\t', all_parameters, parameters[layers[i]].cpu().numpy().shape)
        if 'weight' in layers[i]:
            compute_mask(layers[i],parameters[layers[i]])
            total_weights += np.size(parameters[layers[i]].cpu().numpy())
            zero_weights += (parameters[layers[i]].cpu().numpy()==0).sum()
    print('total parameters: {},\nTotal weights: {},\nZero weights: {},\nZero weights rate: {},\nZero weights rate in model: {}'.format(
        all_parameters, total_weights, zero_weights, colored(zero_weights/total_weights*100, 'blue'), colored(zero_weights/all_parameters*100, 'green')))
    return all_parameters, total_weights, zero_weights

use_cuda = torch.cuda.is_available()
original_namemodel='checkpoint/Pruned_woL2/VGG19_BN_10'
net = Load_model(original_namemodel+'_Pruned24')
CountZeroWeights(net)
CountAllWeights(net)
xxx


iterations = len(train_loss)
all_parameters = np.array(all_parameters)
zero_weights = np.array(zero_weights)
remain_weights = all_parameters-zero_weights


font = {'family' : 'normal',        'weight' : 'bold',        'size'   : 15}
fig, ax1 = plt.subplots(figsize=(15,10))
plt.title('Compression/Accuracy Curve')
# ax1.set_ylim(0.12, 0.17)
ax1.plot(range(iterations+1), all_parameters/remain_weights, '-or', label='Compression Rate')
ax1.set_xlabel('Iterations', color='k')
ax1.set_ylabel('Compression', color='r')


ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='b')
ax2.plot(range(iterations+1), accuracies, '-ob', label='Accuracy')

handles, labels = ax1.get_legend_handles_labels()
handles1, labels1 = ax2.get_legend_handles_labels()
handles.extend(handles1)
labels.extend(labels1)
ax2.set_ylim(91.5, 93)
ax1.legend(handles, labels,loc="upper left")
plt.show()



plt.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(15,10))
plt.title('Compression/Accuracy Curve')
ax1.set_ylim(0.1, 1.)
ax1.plot(range(iterations+1), remain_weights/all_parameters, '-or', label='Weights rate')
ax1.set_xlabel('Iterations', color='k')
ax1.set_ylabel('Non-zero Weights rate (%)', color='r')


ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='b')
ax2.plot(range(iterations+1), accuracies, '-ob', label='Accuracy')

handles, labels = ax1.get_legend_handles_labels()
handles1, labels1 = ax2.get_legend_handles_labels()
handles.extend(handles1)
labels.extend(labels1)
ax2.set_ylim(91.5, 93)
ax1.legend(handles, labels)
plt.show()



plt.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(15,10))
plt.title('Compression/Accuracy Curve')
ax1.set_ylim(0.11, 0.17)
ax1.plot(range(iterations+1), remain_weights/all_parameters, '-or', label='Weights rate')
ax1.set_xlabel('Iterations', color='k')
ax1.set_ylabel('Non-zero Weights rate (%)', color='r')


ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='b')
ax2.plot(range(iterations+1), accuracies, '-ob', label='Accuracy')

handles, labels = ax1.get_legend_handles_labels()
handles1, labels1 = ax2.get_legend_handles_labels()
handles.extend(handles1)
labels.extend(labels1)
ax2.set_ylim(91.5, 93)
ax1.legend(handles, labels)
plt.show()








###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
def Read_training(cluster_name):
    original_namemodel='checkpoint/'+cluster_name+'/VGG19_BN_10'
    train_iter_data = pickle.load(gzip.open('./'+original_namemodel+'_Shared'+'.p.gz', 'rb'))
    train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, unique_weights = train_iter_data
    iterations = len(train_loss)
    all_parameters = np.array(all_parameters)
    unique_weights = np.array(unique_weights)
    remain_weights = unique_weights
    return train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, unique_weights



# test(0,'no')
original_namemodel='checkpoint/Pruned_woL2/VGG19_BN_10'
net = Load_model(original_namemodel+'_Pruned24')
a,b,c = CountZeroWeights(net)
CountAllWeights(net)

clusters_list = [0, 4,5,7,9,  11,13,15,  18,20,25,35]

original_namemodel='checkpoint/Kmeans/VGG19_BN_10'
net = Load_model(original_namemodel+'_Shared35')
CountZeroWeights(net)
CountAllWeights(net)


train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, unique_weights = Read_training('Kmeans')
font = {'family' : 'normal',        'weight' : 'bold',        'size'   : 15}
plt.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(15,10))
ax2 = ax1.twinx()
plt.title('Compression/Accuracy Curve')
# ax1.set_ylim(0.12, 0.17)
ax1.plot(clusters_list, all_parameters/unique_weights, '--or', label='Kmeans Compression Rate')
ax1.set_xlabel('Number of Clusters per layer', color='k')
ax1.set_ylabel('Compression', color='r')
ax1.set_xticks(clusters_list)
ax1.set_xlim(0, 35.2)

ax2.set_ylabel('Accuracy', color='b')
ax2.plot(clusters_list, accuracies, '--ob', label='Accuracy Kmeans')



train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, unique_weights = Read_training('GMM')
ax1.plot(clusters_list, all_parameters/unique_weights, '-or', label='GMM Compression Rate')
ax1.set_xlabel('Number of Clusters per layer', color='r')
ax1.set_ylabel('Compression', color='r')
ax1.set_xticks(clusters_list)
ax1.set_xlim(0, 35.2)

ax2.set_ylabel('Accuracy', color='b')
ax2.plot(clusters_list, accuracies, '-ob', label='Accuracy GMM')

handles, labels = ax1.get_legend_handles_labels()
handles1, labels1 = ax2.get_legend_handles_labels()
handles.extend(handles1)
labels.extend(labels1)
ax2.set_ylim(88, 93.5)
ax2.set_xlim(0, 35.2)
ax1.legend(handles, labels, loc="best")
plt.show()


train_loss, train_acc, test_loss, test_acc, description, accuracies1, all_parameters, total_weights, unique_weights1 = Read_training('Non_parametric1')
train_loss, train_acc, test_loss, test_acc, description, accuracies2, all_parameters, total_weights, unique_weights2 = Read_training('Non_parametric2')
train_loss, train_acc, test_loss, test_acc, description, accuracies3, all_parameters, total_weights, unique_weights3 = Read_training('Non_parametric3')


accuracies = [accuracies1[0], accuracies1[1],  accuracies2[-1], accuracies3[-1]]
unique_weights = [unique_weights1[0], unique_weights1[1],  unique_weights2[-1], unique_weights3[-1]]

plt.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(15,10))
plt.title('Compression/Accuracy Curve')
ax1.set_ylim(0, 1.)
ax1.plot(range(len(accuracies)), unique_weights/all_parameters, '-or', label='Weights rate')
ax1.set_xlabel('Covariance Initialization', color='k')
ax1.set_ylabel('Unique non-zero Weights rate (%)', color='r')
ax1.set_xticks([0,1,2,3])


ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='b')
ax2.plot(range(len(accuracies)), accuracies, '-ob', label='Accuracy')

handles, labels = ax1.get_legend_handles_labels()
handles1, labels1 = ax2.get_legend_handles_labels()
handles.extend(handles1)
labels.extend(labels1)
ax2.set_ylim(91., 93.5)
ax1.legend(handles, labels)
plt.show()


plt.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(15,10))
plt.title('Compression/Accuracy Curve')
ax1.set_ylim(0., 0.000015)
ax1.plot(range(len(accuracies)), unique_weights/all_parameters, '-or', label='Weights rate')
ax1.set_xlabel('Covariance Initialization', color='k')
ax1.set_ylabel('Non-zero Weights rate (%)', color='r')
ax1.set_xticks([0,1,2,3])


ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='b')
ax2.plot(range(len(accuracies)), accuracies, '-ob', label='Accuracy')

handles, labels = ax1.get_legend_handles_labels()
handles1, labels1 = ax2.get_legend_handles_labels()
handles.extend(handles1)
labels.extend(labels1)
ax2.set_ylim(91., 93.5)
ax1.legend(handles, labels)
plt.show()




fig, ax1 = plt.subplots(figsize=(15,10))
plt.title('Error Model Curve')

# ax1.set_ylim(0.12, 0.17)
ax1.plot(range(len(test_acc)), 100-np.array(train_acc), '--b', label='Training')
ax1.plot(range(len(test_acc)), 100-np.array(test_acc), '-r', label='Test')
ax1.set_xlabel('Number of epochs', color='k')
ax1.set_ylabel('Error', color='r')

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)
plt.show()


train_loss, train_acc, test_loss, test_acc, description, accuracies, all_parameters, total_weights, unique_weights = Read_training('Kmeans')
fig, ax1 = plt.subplots(figsize=(15,10))
plt.title('Error Model Curve')

# ax1.set_ylim(0.12, 0.17)
ax1.plot(range(test_acc.shape[1]), 100-np.array(train_acc).T, '--b', label='Training')
ax1.plot(range(test_acc.shape[1]), 100-np.array(test_acc).T, '-r', label='Test')
ax1.set_xlabel('Number of epochs', color='k')
ax1.set_ylabel('Error', color='r')

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::int(len(test_acc))], labels[::int(len(test_acc))])
plt.show()










# # Maybe S3Pool
# fig, ax1 = plt.subplots(figsize=(15,10))
# plt.title('Error Model Curve')

# # ax1.set_ylim(0.12, 0.17)
# ax1.plot(range(len(test_acc)), 100-np.array(test_acc), '--b', label='Training')
# ax1.plot(range(len(test_acc)), 100-np.array(train_acc), '-r', label='Test')
# ax1.set_xlabel('Number of epochs', color='k')
# ax1.set_ylabel('Accuracy', color='r')

# ax2 = ax1.twinx()
# ax2.set_ylabel('Weights L1 values', color='k')
# ax2.plot(range(len(test_acc)), np.array(total_weights[:-1]), '-k', label='Weights sum')

# handles, labels = ax1.get_legend_handles_labels()
# handles1, labels1 = ax2.get_legend_handles_labels()
# handles.extend(handles1)
# labels.extend(labels1)
# ax1.legend(handles, labels)
# plt.show()






