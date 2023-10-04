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
with open('POLY_parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(gamma1,regScale1)] = parametersAndRegScales

kPoly1 = lambda X1,X2: kernelPoly(X1,X2,gamma1)
clf1=kC.KernelClassifier(kPoly1)

with open('RBF_parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(gamma2,regScale2)] = parametersAndRegScales

kRBF1 = lambda X1,X2: kernelRBF(X1,X2,gamma2)
clf2=kC.KernelClassifier(kRBF1)

with open('Standardized_POLY_parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(gamma3,regScale3)] = parametersAndRegScales

kPoly2 = lambda X1,X2: kernelPoly(X1,X2,gamma3)
clf3=kC.KernelClassifier(kPoly2,withStandardization=True)

with open('Standardized_RBF_parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(gamma4,regScale4)] = parametersAndRegScales

kRBF2 = lambda X1,X2: kernelRBF(X1,X2,gamma4)
clf4=kC.KernelClassifier(kRBF2,withStandardization=True)

kernelsAndRegScales = [(clf1,regScale1),(clf2,regScale2),(clf3,regScale3),(clf4,regScale4)]

# 2
(generatingParameters,obsData) = np.load('ObservedData.npy')


#--------Run the ratio estimation algorithm -----------------------------------------------------------------------#
# values at which to compute the posterior
mu_values = np.linspace(-6,6,100)
Estimator = PosteriorEstimator(randomSeed)
estimated_probabilities = Estimator.estimateProbabilities(obsData, mu_values, model, n_samples, kernelsAndRegScales)

with open('estimated_probabilities.pickle', 'wb') as f:
    pickle.dump((mu_values,estimated_probabilities), f)
  