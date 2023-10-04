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
regScale = np.linspace(1e-2,1e12,40)

kPoly1 = lambda X1,X2: kernelPoly(X1,X2,1)
clf1=kC.KernelClassifier(kPoly1)

kPoly3 = lambda X1,X2: kernelPoly(X1,X2,3)
clf3=kC.KernelClassifier(kPoly3)

kPoly5 = lambda X1,X2: kernelPoly(X1,X2,5)
clf5=kC.KernelClassifier(kPoly5)

kernelsAndRegScales = [(clf1,regScale),(clf3,regScale),(clf5,regScale)]

# 2
(generatingParameters,obsData) = np.load('ObservedData.npy')
obsData = np.vstack([0,obsData])

#--------Run the ratio estimation algorithm -----------------------------------------------------------------------#
# values at which to compute the posterior
mu_values = np.linspace(-6,6,100)[:20]
Estimator = PosteriorEstimator(randomSeed)
estimated_probabilities = Estimator.estimateProbabilities(obsData, mu_values, model, n_samples, kernelsAndRegScales)

with open('estimated_probabilitiesH1.pickle', 'wb') as f:
    pickle.dump((mu_values,estimated_probabilities), f)
  