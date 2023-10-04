import pickle
import numpy as np
import Utilities.kernelClassifier as kC
from Utilities.kernels import kernelLinear, kernelPoly, kernelRBF
from Utilities.LFIRE_Estimator import PosteriorEstimator
from Utilities.ExpModels import  GaussianModel

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model = GaussianModel(3,-20,20, randomSeed) 
#Number of samples to be created by each generator for classification.
n_samples=1000


#-------Load computed values from other scripts -------------------------------------------------------------------#
# 1
with open('parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(gamma,regScale)] = parametersAndRegScales

kPoly = lambda X1,X2: kernelPoly(X1,X2,gamma)
clf=kC.KernelClassifier(kPoly)

kernelsAndRegScales = [(clf,regScale)]

# 2
(generatingParameters,obsData) = np.load('ObservedData.npy')


#--------Run the ratio estimation algorithm -----------------------------------------------------------------------#
# values at which to compute the posterior
mu_values = np.linspace(-6,6,100)
Estimator = PosteriorEstimator(randomSeed)
estimated_probabilities = Estimator.estimateProbabilities(obsData, mu_values, model, n_samples, kernelsAndRegScales)

with open('estimated_probabilities.pickle', 'wb') as f:
    pickle.dump((mu_values,estimated_probabilities), f)
  