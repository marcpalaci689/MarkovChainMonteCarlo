'''
@autor: Marc Palaci-Olgun
@date : 10/5/2017

Test PDFs to test the MCMC algorithms 
'''

import numpy as np

def gaussian_mixture_1d(x):
    
    '''
    Mixture of 1d Gaussians. The number of Gaussians to mix is num and the mean and var arrays accept the 
    mean and variance of each mixture. The weights array accepts the weights of each component. 
    '''
    
    weights=[0.4,0.6] # Mixture weights used.
    assert(sum(weights)==1.0), "mixture weights must sum to 1"
       
    mean=[-2,4] # means of Gaussians. Alter if you wish to shift the Gaussians
    var=[2,1]   # variances of Gaussians. Alter if you wish to change the width of the Gaussian
    
    
    P = 0    
    for i in xrange(len(weights)):
        v  = var[i]
        m  = mean[i]
        P += weights[i]*(((2*np.pi)**(-1.0/2.0))*(v)**(-0.5))*np.exp(-(0.5/v)*(x-m)**2)
    return P

def gaussian_mixture_2d(x,weights=[0.4,0.6],mean=[-2,4],var=[2,1]):

    weights=[0.4,0.6] # Mixture weights used.
    assert(sum(weights)==1.0), "mixture weights must sum to 1"
       
    mean=np.array([[-1,2],[3,-3]]) # means of Gaussians. Alter if you wish to shift the Gaussians
    var=np.array([[1,1],[1,2]])   # variances of Gaussians. Alter if you wish to change the width of the Gaussian
   
    P=0 
    for i in xrange(len(weights)):
        v                = var[i,:]
        m                = mean[i,:].reshape(-1,1)
        normalizer       = ((2*np.pi)**(-2.0/2.0))*(np.prod(v)**(-0.5))
        inverse_variance = np.diag(1.0/v)
        P               += np.asscalar(normalizer*np.exp(-0.5*(x-m).T.dot(inverse_variance).dot(x-m)))
    return P    

def G(x):
    variance = np.array([[3,0],[0,3]])
    mean     = np.array([4,2]).reshape(-1,1) 
    normalizer = ((2*np.pi)**(-2.0/2.0))*(np.prod(np.diag(variance))**(-0.5))
    inverse_variance = np.diag(1.0/np.diag(variance))
    return np.asscalar(normalizer*np.exp(-0.5*(x-mean).T.dot(inverse_variance).dot(x-mean)))    

def N(x):
    variance = 1
    mean     = 0 
    normalizer = ((2*np.pi)**(-1.0/2.0))*(variance)**(-0.5)
    return normalizer*np.exp(-(0.5/variance)*(x-mean)**2)