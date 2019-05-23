import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from termcolor import colored, cprint
from itertools import cycle, islice

from deep_utils import *

def Plotting(X,colors,centroids):
    Xt = X.numpy()
    yt = colors.numpy()
    centroids_plot = centroids.numpy()

    # first plot
    # plt.hold(True)
    # plt.scatter(X.numpy(), [0]*n, c=c_i.numpy()[:,None], s=120)
    # plt.scatter(centroids.numpy(), [0]*num_clusters, c=['red']*num_clusters, s=150, marker='*')
    # plt.hold(False)
    # plt.show()

    # second plot
    axisplot=500
    minX, maxX = np.floor(Xt.min()), np.ceil(Xt.max())
    bins_plot = np.linspace(minX, maxX, axisplot)
    hist, bins = np.histogram(Xt, bins=bins_plot)
    minY, maxY = hist.min(), hist.max()


    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(yt) + 1))))
    width = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2
    hist[np.abs(center).min()==np.abs(center)]=0
    color = []
    for p in range(len(hist)):
        c = yt[np.logical_and(Xt>bins[p] , Xt<bins[p+1])]
        if np.size(c)==0:
            color.append(0)
            # print('-')
        else:
            color.append( int(np.round(c.mean())) )
            # print(color[-1])
    plt.hold(True)
    plt.bar(center, hist, align='center', width=width, color=colors[color])
    # plt.stem(clusters, [hist.min()]*len(clusters), markerfmt=' ', linefmt='ko-')
    # plt.bar(clusters, [hist.max()]*length, align='center', width=[width[0]/2]*length, color='k'*length) # color=colors[np.unique(y_pred)]
    plt.plot(centroids_plot, [-1]*len(centroids_plot), 'ko', markersize=10)
    plt.hold(False)
    plt.xticks(np.arange(minX, maxX , (maxX-minX)/10) )
    plt.yticks(np.arange(minY, maxY+10 , 15 ))
    plt.ylim(minY-5, maxY+10)
    plt.show()

# this code implements pruning using register buffer feature to save input mask
def compute_mask(layer, weights):
    thresh = weights.std()
    m1 = weights.abs() < thresh
    mask = torch.ones(weights.size()).cuda()
    mask = mask-m1.float()
    print('Layer:\t',layer, '\tThreshold:\t', thresh,'\t\tPrunned weights (%):\t', colored((1-mask.mean())*100,'red') )
    return mask

class SharingWeightsNetwork(nn.Module):
    def __init__(self, model,cluster=10):
        super(SharingWeightsNetwork, self).__init__()
        # get the model ready to be pruned.
        self.base_model = model
        self.all_layers = list(self.base_model.state_dict().keys())
        self.num_clusters = cluster
        self.list_clusters = []
        self.modules = []
        for i in range(len(self.all_layers)):
            self.modules.append(self.all_layers[i].split('.')[:-1])

    def Sharing(self,method):
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

                # compute the mask
                if method=='kmeans':
                    weights, num_clusters = clustering_method_1(self.all_layers[layer], weights,self.num_clusters)
                elif method=='GMM':
                    weights, num_clusters = clustering_method_2(self.all_layers[layer], weights,self.num_clusters)
                elif method=='GMM-kmeans':
                    weights, num_clusters = clustering_method_3(self.all_layers[layer], weights,self.num_clusters)
                self.list_clusters.append(num_clusters)
                current_layer.weight.data =  weights

    def forward(self, x):
        return self.base_model(x)



############################
#### Clustering methods ####
############################
def clustering_method_1(layer, weights, num_clusters, iterations=50): # kmeans
    index = weights!=0
    X = weights[index]
    n = len(X)

    if n>=num_clusters:
        if num_clusters>1:
            ind = torch.round(torch.linspace(0, n-1 ,num_clusters))
        else:
            ind = torch.FloatTensor([n/2])
        Xi, _ = X.sort()
        centroids = Xi[ind.cuda().long()]
        centroidsI = centroids
        for it in np.arange(0,iterations):
            distances = (X[:,None]-centroids).abs()
            _, c_i = distances.min(dim=1)

            # Plotting(X,c_i,centroids)
            for i in range(num_clusters):
                centroids[i:i+1] = (X[c_i==i]).sum(0) / (c_i==i).sum()
        weights[index] = centroids[c_i]
        print('Points: ',n, '  Initial Kmeans: ', len(centroidsI), '   Final Kmeans: ', len(np.unique(c_i.cpu().numpy())))
    else:
        num_clusters = n
        print('Points: ',n, 'Initial Kmeans: ', num_clusters, '   Final Kmeans: ', n)
    return weights, num_clusters

def clustering_method_2(layer, weights, num_clusters, iterations=50): # GMM
    # Main funtion
    index = weights!=0
    X = weights[index]
    n = len(X)
    if n>=num_clusters:
        it, phi, u = Constrained_GMM(X*1e12, num_clusters)
        _, c_i = phi.max(1)
        unique_cluster = CudaPytorchunique(c_i)
        weights[index] = u[c_i]/1e12
        print('iteration:  ', it, '   Points: ',n, '  Initial Gaussian Mixture: ', num_clusters, '   Final Gaussian Mixture: ', len(CudaPytorchunique(c_i)))
    else:
        num_clusters = n
        print('iteration:    -- ', '   Points: ',n, '   Initial Gaussian Mixture: ', num_clusters, '   Final Gaussian Mixture: ', n)
    return weights, num_clusters


def clustering_method_3(layer, weights, num_clusters, iterations=10): #Kmeans using GMM as cluster number initializer.
    # Main funtion
    index = weights!=0
    X = weights[index]
    n = len(X)
    if n>=num_clusters:
        # covariance_scale= X.std()/(torch.median(X) * n) # Heuristic value
        covariance_scale = 1e7 #1e6 -> 1e8
        # print(colored([torch.median(X), X.std(), '               ', covariance_scale],'blue'))
        it, phi, u = GMM(X*1e12, num_clusters,covariance_scale,iterations)
        _, c_i = phi.max(1)
        num_clusters = len(CudaPytorchunique(c_i))
        weights, num_clusters = clustering_method_1(layer, weights, num_clusters, iterations=50)
    else:
        num_clusters = n
        print('iteration:    -- ', '   Points: ',n, '   Initial number of clusters: ', num_clusters, '   Final number of clusters: ', n)
    return weights, num_clusters




################################
# Utils for clustering methods #
################################
def GaussianDistribution(X, ui, Di):
    if Di[0]<1e-8:
        Di=torch.cuda.FloatTensor([1e-8])
    exponential_term = torch.exp(-0.5 *    (X-ui)*(X-ui) / Di   )
    return exponential_term / torch.sqrt(Di)

def Expectation_step(X, phi, u, D, pi, k):
    for i in range(0,k):
        phi[:,i] = pi[i:i+1]*GaussianDistribution(X,u[i:i+1],D[i:i+1])

    suma = phi.sum(-1)
    suma[suma<torch.cuda.FloatTensor([1e-24])] = 1e-24
    phi = phi/suma[:,None]
    return phi
def MaximumLikelihood_step(X, k, u, D,phi):
    nk = phi.sum(0)
    pi = nk/nk.sum()
    for ki in range(k):
        u[ki:ki+1] = (phi[:,ki] * X).sum(0) / nk[ki]
        D[ki:ki+1] = (phi[:,ki] * (X - u[ki:ki+1]) * (X - u[ki:ki+1]) ).sum() / nk[ki]
    return u, D, pi

def Initialization(X,k,covariance_scale=1):
    n = len(X)
    ind = torch.round(torch.linspace(0, n-1 ,k))
    Xi, _ = X.sort()
    u = Xi[ind.cuda().long()]

    D = torch.cuda.FloatTensor([(torch.mean(X*X) - torch.mean(X)**2)/(k*2)*covariance_scale]).repeat(k)
    # D = torch.cuda.FloatTensor([0.01]).repeat(k)
    pi =  torch.cuda.FloatTensor([1./k]).repeat(k)
    phi = torch.zeros([n,k]).cuda()
    return u, D, pi, phi, X

def GMM(X, num_clusters,covariance_scale=1, iterations=50):
    u, D, pi, phi, X = Initialization(X, num_clusters,covariance_scale) # Initialize parameters and set outliers with the mean value in each position.
    for it in np.arange(0,iterations):
        # E-step
        phi = Expectation_step(X, phi, u, D, pi, num_clusters)
        # M-step
        u, D, pi = MaximumLikelihood_step(X, num_clusters, u, D, phi)
    return it, phi, u

def Constrained_GMM(X, num_clusters,covariance_scale=1, iterations=50): # Constrain on number of cluster assigned by this method.
    u, D, pi, phi, X = Initialization(X, num_clusters, covariance_scale) # Initialize parameters and set outliers with the mean value in each position.
    uI = u
    for it in np.arange(0,iterations):
        # E-step
        phi = Expectation_step(X, phi, u, D, pi, num_clusters)
        # Constrain on number of cluster assigned by this method.
        if len(CudaPytorchunique(phi.max(1)[1]))!=num_clusters:
            phi = temp.clone();   break;
        # M-step
        u, D, pi = MaximumLikelihood_step(X, num_clusters, u, D, phi)
        temp = phi.clone()
    return it, phi, u