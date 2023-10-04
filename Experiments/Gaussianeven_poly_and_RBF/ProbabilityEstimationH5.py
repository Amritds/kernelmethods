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
with open('POLY_deg2.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
regScale2 = parametersAndRegScales

kPoly2 = lambda X1,X2: kernelPoly(X1,X2,2)
clf2=kC.KernelClassifier(kPoly2)

# 4
with open('POLY_deg4.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
regScale4 = parametersAndRegScales

kPoly4 = lambda X1,X2: kernelPoly(X1,X2,4)
clf4=kC.KernelClassifier(kPoly4)

# 6
with open('POLY_deg6.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
regScale6 = parametersAndRegScales

kPoly6 = lambda X1,X2: kernelPoly(X1,X2,6)
clf6=kC.KernelClassifier(kPoly6)

# 6
with open('POLY_deg6.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
regScale6 = parametersAndRegScales

kPoly6 = lambda X1,X2: kernelPoly(X1,X2,6)
clf6=kC.KernelClassifier(kPoly6)

#RBF
with open('RBF_parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(gamma,regScaleRBF)] = parametersAndRegScales

kRBF = lambda X1,X2: kernelRBF(X1,X2,gamma)
clfRBF=kC.KernelClassifier(kRBF)

kernelsAndRegScales = [(clf2,regScale2),(clf4,regScale4),(clf6,regScale6),(clfRBF,regScaleRBF)]

# 2
(generatingParameters,obsData) = np.load('ObservedData.npy')
obsData = np.vstack([0,obsData])

#--------Run the ratio estimation algorithm -----------------------------------------------------------------------#
# values at which to compute the posterior
mu_values = np.linspace(-6,6,100)[80:100]
Estimator = PosteriorEstimator(randomSeed)
estimated_probabilities = Estimator.estimateProbabilities(obsData, mu_values, model, n_samples, kernelsAndRegScales)

with open('estimated_probabilitiesH5.pickle', 'wb') as f:
    pickle.dump((mu_values,estimated_probabilities), f)
  