import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.stats import norm, entropy

########################################### Generic functions #########################################################################
#Calculates kl divergence for multiple observed data values.
def kLDiv(true_posterior,estimated_posterior):
    # input: n_obsData x calculated true posterior at para_values
    #        n_obsData x estimated posterior at para_values
    #
    # output: n_obsData x K.L divergence
    
    samples = np.dstack((true_posterior,estimated_posterior))
        
    divergence =entropy(samples[0,:,0],samples[0,:,1])
    for x in samples[1:]:
        divergence = np.hstack([divergence,entropy(x[:,0],x[:,1])])
    return divergence


#Weighted average of theta values.
def expectedMean(theta_values,estimated_posterior):
    # input: k x r_para_values
    #        k_estimated posteriors at para_values
    #
    # output: r expected para_values
    estimated_posteriorReshape = estimated_posterior.reshape((1,len(estimated_posterior)))
    return np.dot(estimated_posteriorReshape,theta_values)[0]
    
#Relative error for multiple observed data.
def relativeError(theta_values,estimated_posteriorS,generating_theta_values):
    # input: k_para_values
    #        n_obsData estimated posteriors
    #        n x r_obsData generating para_values
    #
    # output: n x r _obsData relative errors
    
    expectedMeans = expectedMean(theta_values,estimated_posteriorS[0])
    for estimated_posterior in estimated_posteriorS[1:]:
        expectedMeans = np.vstack([expectedMeans,expectedMean(theta_values,estimated_posterior)])

    relativeErrors  = np.sqrt(np.power(expectedMeans - generating_theta_values,2)/np.power(generating_theta_values,2))
    return relativeErrors



########################################### Gaussian Model ############################################################################

class GaussianModel:
    def __init__(self,sigma,minRange,maxRange,randomSeed=432):
        #Sample from gaussian width 3 and assume a uniform dist over mu values (-minRange,maxRange) for the marginal
        self.minRange =minRange
        self.maxRange =maxRange
        self.sigma = sigma
        self.prior = 1/(maxRange-minRange)
        np.random.seed(randomSeed)
    
    #Generates n 1-D samples from a gaussian ditribution with given mean.
    def generator_paraGiven(self,mu,n_samples):
        X = np.random.normal(mu,self.sigma,(n_samples,1))
        return X

    #Generates n 1-D samples from gaussian ditributions, assuming a uniform dist.
    def generator_marginal(self,n_samples):
        mu= np.random.uniform(self.minRange,self.maxRange,(n_samples,1));
        X = np.random.normal(mu,self.sigma,(n_samples,1));
        return X
    
    #Generates n 1-D samples from gaussian ditributions, assuming a uniform dist, return the generating parameres.
    def generate_ObsData(self,n_samples):
        mu= np.random.uniform(self.minRange,self.maxRange,(n_samples,1));
        X = np.random.normal(mu,self.sigma,(n_samples,1));
        return (mu,X)
    
    
    #Generates the true posterior at given points
    #Calculates for a single observed_data value.
    def truePosterior(self,mu_values,observed_data):
        sigma=self.sigma
        a_0= lambda mu: -(mu*mu)/(2*sigma*sigma)-np.log(np.sqrt(2*np.pi)*sigma)-np.log(norm.cdf((self.maxRange-mu)/sigma)-norm.cdf((self.minRange-mu)/sigma))
        a_1= lambda mu: mu/(sigma*sigma)
        a_2= lambda mu: -1/(2*sigma*sigma)

        true_probabilities= np.zeros(len(mu_values))
        i=0
        for mu in mu_values:
            true_probabilities[i] = np.exp(a_0(mu)+a_1(mu)*observed_data+a_2(mu)*np.power(observed_data,2))
            i=i+1 
        return true_probabilities
    
    #Calculates kl divergence for multiple observed data values.
    def klDivergence(self,mu_values,observed_data,estimated_posterior):
        true_posterior = np.apply_along_axis(lambda obsData: self.truePosterior(mu_values,obsData), 1, observed_data)
        return kLDiv(true_posterior,estimated_posterior)

#################################### Arch Model #######################################################################################

class ArchModel:
    def __init__(self,theta1_minRange,theta1_maxRange,theta2_minRange,theta2_maxRange,randomSeed=432):
        #Sample from ARCH(1) model and assume a uniform dist over theta values (-minRange,maxRange) for the marginal
        self.theta1_minRange =theta1_minRange
        self.theta1_maxRange =theta1_maxRange
        
        self.theta2_minRange =theta2_minRange
        self.theta2_maxRange =theta2_maxRange
        
        self.theta1_prior = 1/(theta1_maxRange-theta1_minRange)
        self.theta2_prior = 1/(theta2_maxRange-theta2_minRange)
        
        self.prior = self.theta1_prior * self.theta2_prior
        
        np.random.seed(randomSeed)
    
    def GeneratorThetaGiven(self,theta_values, n_samples,e=None,epsilon=None,T=100):
        theta_1=theta_values[0]
        theta_2=theta_values[1]
        
        # T timesteps
        
        #y_0 = 0 for all samples
        y_0 = np.zeros((n_samples,1))
    
        #initialize independent random variables (as required execpt in testing)
        if e is None:
            e = np.random.normal(0,1,n_samples)
        if epsilon is None:
            epsilon = np.random.normal(0,1,(n_samples,T))

        #Calculate y_1, e_1 for all samples
        e = epsilon[:,0]*np.sqrt(0.2 + theta_2* np.power(e,2)) 
        y = e.reshape((n_samples,1))
        
        #Save e vector for simple true posterior computation.
        self.e_vector = e.reshape(n_samples,1)
        
        #Calculate y, e for remaining time steps
        for i in range(1,T):
            e = epsilon[:,i]*np.sqrt(0.2 + theta_2 * np.power(e,2)) 
            y_t = (theta_1 * y[:,-1] + e).reshape(n_samples,1)
            y=np.hstack([y, y_t])
            self.e_vector = np.hstack([self.e_vector,e.reshape((n_samples,1))])
            
            #preprocessing
            y = y/np.sqrt(T)
            
        return y

    def GeneratorMarginal(self,n_samples):
        theta_1 = np.random.uniform(self.theta1_minRange,self.theta1_maxRange ,n_samples)
        theta_2 = np.random.uniform(self.theta2_minRange,self.theta2_maxRange ,n_samples)
        thetas = np.dstack([theta_1,theta_2])[0].T
        print(thetas.shape)
        return (thetas,self.GeneratorThetaGiven(thetas,n_samples))

    #Return summary stats for n_samples of y vectors.
    def SummaryStats(self,y):
        result = np.correlate(y, y, mode='full')
        a = int(result.size/2)
        return result[a:a+5]
    
    #Generates n summarystats of samples from an ARCH(1) model with given parameters.
    def generator_paraGiven(self,theta_values,n_samples):
        X=np.apply_along_axis(self.SummaryStats, 1, self.GeneratorThetaGiven(theta_values, n_samples))
        return X

    #Generates n summarystats of samples from an ARCH(1) model for parameters over theta_1 from [-1,1] and theta_2 from [0,1].
    def generator_marginal(self,n_samples):
        (thetas, marginal_samples) = self.GeneratorMarginal(n_samples)
        X=np.apply_along_axis(self.SummaryStats, 1, marginal_samples)
        return X   
    
    #Generates n summarystats of samples from an ARCH(1) model for parameters over theta_1 from [-1,1] and theta_2 from [0,1].
    def generate_ObsData(self,n_samples):
        (thetas, marginal_samples) = self.GeneratorMarginal(n_samples)
        X=np.apply_along_axis(self.SummaryStats, 1, marginal_samples)
        return (thetas,X)   
    
    
############### Dynamic bistable HMM - https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/A_dynamic_bistable_hidden_Markov_model.svg/326px-A_dynamic_bistable_hidden_Markov_model.svg.png #######################################################################################################################################

class HMM_stringGenerator:
    def __init__(self,theta_minRange,theta_maxRange,gamma_minRange,gamma_maxRange,randomSeed=432):
        #Sample from HMM model and assume a uniform dist over theta and gamma values (-minRange,maxRange) for the marginal
        self.theta_minRange =theta_minRange
        self.theta_maxRange =theta_maxRange
        
        self.gamma_minRange =gamma_minRange
        self.gamma_maxRange =gamma_maxRange
        
        self.theta_prior = 1/(theta_maxRange-theta_minRange)
        self.gamma_prior = 1/(gamma_maxRange-gamma_minRange)
        
        self.prior = self.theta_prior * self.gamma_prior
        
        np.random.seed(randomSeed)
    
    #Generates n strings from the HMM with given parameters.
    def generator_paraGiven(self,para_values,n_samples):
        #Model parameters
        theta = para_values[0]
        gamma = para_values[1]
        T =100
        states =np.array(['A','B'])

        #Non deterministic choices of the generator
        timestepOutcomes_theta = np.random.uniform(0,1,(T,n_samples)) <= theta
        timestepOutcomes_gamma = np.random.uniform(0,1,(T,n_samples)) <= gamma

        #Start at state A
        currentState  = np.zeros(n_samples)

        #Generate strings
        strings = np.tile('',n_samples)
        for i in range(0,T):
            currentState = (currentState + timestepOutcomes_theta[i])%2
            error = timestepOutcomes_gamma[i]
            finalSelection = ((currentState + (1-error))%2).astype(int)
            nextStates = np.array(states[finalSelection])
            strings = np.core.defchararray.add(strings,nextStates)
    
        return strings

    #Generates n strings from the HMM for parameters uniformly sampled from given ranges.
    def generator_marginal(self,n_samples):
        theta= np.random.uniform(self.theta_minRange,self.theta_maxRange,(1,n_samples));
        gamma= np.random.uniform(self.gamma_minRange,self.gamma_maxRange,(1,n_samples));
        para_values = np.dstack([theta,gamma])[0].T
        return self.generator_paraGiven(para_values,n_samples)
    
     #Generates n strings from the HMM for parameters uniformly sampled from given ranges.
    def generate_ObsData(self,n_samples):
        theta= np.random.uniform(self.theta_minRange,self.theta_maxRange,(1,n_samples));
        gamma= np.random.uniform(self.gamma_minRange,self.gamma_maxRange,(1,n_samples));
        para_values = np.dstack([theta,gamma])[0].T
        
        return (vals,self.generator_paraGiven(para_values,n_samples))
