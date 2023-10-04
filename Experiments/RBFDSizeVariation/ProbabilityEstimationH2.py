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



#-------Load computed values from other scripts -------------------------------------------------------------------#
#RBF
with open('RBF_parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(gamma,regScaleRBF)] = parametersAndRegScales

kRBF = lambda X1,X2: kernelRBF(X1,X2,gamma)
clfRBF=kC.KernelClassifier(kRBF)

kernelsAndRegScales = [(clfRBF,regScaleRBF)]

# 2
(generatingParameters,obsData) = np.load('ObservedData.npy')
obsData = np.vstack([0,obsData])

#--------Run the ratio estimation algorithm -----------------------------------------------------------------------#
# values at which to compute the posterior
mu_values = np.linspace(-6,6,100)[20:40]
Estimator = PosteriorEstimator(randomSeed)

#Number of samples to be created by each generator for classification.
size_values = np.array([100,200,500,1000,1500,2000,2500])

for n in size_values:
    estimated_probabilities = Estimator.estimateProbabilities(obsData, mu_values, model, n, kernelsAndRegScales)
    with open('N_'+str(n)+'_estimated_probabilitiesH2.pickle', 'wb') as f:
        pickle.dump((mu_values,estimated_probabilities), f)
  