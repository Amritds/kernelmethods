import pickle
import numpy as np
import Utilities.kernelClassifier as kC
from Utilities.kernels import kernelLinear, kernelPoly, kernelRBF, kernelFourier
from Utilities.LFIRE_Estimator import PosteriorEstimator
from Utilities.ExpModels import  ArchModel

#Random Seed
randomSeed = 432

#-------Load computed values from other scripts -------------------------------------------------------------------#
# 1
with open('RBF_parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(T1,skip1,gamma1,regScale1)] = parametersAndRegScales

kF1 = lambda X1,X2: kernelFourier(X1,X2,gamma1)
clf1=kC.KernelClassifier(kF1)

kernelsAndRegScales = [(clf1,regScale1)]

# 2
with open('observedData.pickle', 'rb') as f:
     observedData = pickle.load(f)
(generatingParameters,obsData) = observedData


#------Exp Setup -------------------------------------------------#
# Model
model = ArchModel(-1,1, 0,1, randomSeed,T= T1,skip= skip1) 
#Number of samples to be created by each generator for classification.
n_samples=1000

#--------Run the ratio estimation algorithm -----------------------------------------------------------------------#
# values at which to compute the posterior
n1=10
n2=10
theta1_values=np.linspace(-0.9,0.9 , n1)
theta2_values=np.linspace( 0.1,0.9 , n2)
xv,yv = np.meshgrid(theta1_values,theta2_values)
theta_vals = np.dstack((xv,yv))
theta_values = theta_vals.reshape(n1*n2,2)

Estimator = PosteriorEstimator(randomSeed)
estimated_probabilities = Estimator.estimateProbabilities(obsData, theta_values, model, n_samples, kernelsAndRegScales)

with open('estimated_probabilities.pickle', 'wb') as f:
    pickle.dump((theta_values,estimated_probabilities), f)
