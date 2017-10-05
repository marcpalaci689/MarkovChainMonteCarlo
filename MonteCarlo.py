'''
@autor: Marc Palaci-Olgun
@date : 10/5/2017

Simple Markov Chain Monte Carlo algorithms. 
'''
import numpy as np
import matplotlib.pyplot as plt
import test_pdf as pdf

def NN(x):
    '''
    Mixture of 2 gaussians
    '''
    variance1   = 2
    mean1       = -1 
    normalizer1 = ((2*np.pi)**(-1.0/2.0))*(variance1)**(-0.5)
    
    variance2   = 3
    mean2       = 6 
    normalizer2 = ((2*np.pi)**(-1.0/2.0))*(variance2)**(-0.5)    
    return 0.4*normalizer1*np.exp(-(0.5/variance1)*(x-mean1)**2) +0.6*normalizer2*np.exp(-(0.5/variance2)*(x-mean2)**2) 

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

class MCMC:
    
    def __init__(self,distribution,dimensionality,algorithm='MH'):
        self.algorithm = algorithm        # Algorithm to use
        self.P         = distribution     # Probability distribution we wish to sample
        self.D         = dimensionality   # Dimensionality of search space
    
    def metropolis_hastings_update(self):
        x_proposal = np.random.normal(loc=self.x,scale=self.step,size=(self.D,1)) # propose a new state
        acceptance = min(1,self.P(x_proposal)/self.P(self.x))                     # Compute acceptance probability
        u = np.random.uniform()                                                   # Sample a random threshold probability
        if u<acceptance:                                                          # If acceptance probability is accepted, update x and history
            self.x = x_proposal
            self.history = np.hstack((self.history,x_proposal))
        return
    
    def run(self,step=2,iterations=2000):
        self.step    = step                                                   # Perturbation stepsize
        self.x       = np.random.normal(scale = self.step,size=(self.D,1))    # Initiate starting point   
        self.history = self.x                                                 # Keep track of all accepted states
        
        # Perform MCMC for the specified number of iteration
        for i in xrange(iterations):
            if self.algorithm == 'MH':
                self.metropolis_hastings_update()   # Use MH updates
    
    def density_plot(self):
        assert(self.D<=2), 'Cannot plot for data of higher dimensionality of 2'
        if self.D==2:
            x = np.linspace(np.min(self.history[0,:])-0.2,np.max(self.history[0,:])+0.2,100)
            y = np.linspace(np.min(self.history[1,:])-0.2,np.max(self.history[1,:])+0.2,100)
            P = np.zeros((100,100))
            for i in xrange(100):
                for j in xrange(100):
                    P[i,j] = self.P(np.array([x[i],y[j]]).reshape(-1,1))
            
            plt.clf()
            plt.contourf(x,y,P.T,50)
            plt.colorbar()                    
            plt.scatter(self.history[0,:],self.history[1,:],s=5,c='w',label='Sampled points')
            plt.title('Probability Density Plot')
            plt.legend()
        
        else:
            x = np.linspace(np.min(self.history)-0.2,np.max(self.history)+0.2,10000)
            P = np.zeros(10000)
            for i in xrange(10000):
                P[i] = self.P(x[i])
            plt.clf()
            plt.plot(x,P)
            plt.scatter(self.history,np.zeros(self.history.shape[1]),s=5,c='r',label='Sampled points')
            plt.title('Probability Density Plot')
            plt.legend()            
                

if __name__ == '__main__':
    x = np.array([0,0]).reshape(-1,1)
    p=pdf.gaussian_mixture_2d(x)
    
    MCMC = MCMC(pdf.gaussian_mixture_2d,dimensionality=2)
    MCMC.run()
    
