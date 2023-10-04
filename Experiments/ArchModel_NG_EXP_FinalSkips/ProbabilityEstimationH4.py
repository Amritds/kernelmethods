import pickle
import numpy as np
import Utilities.kernelClassifier as kC
from Utilities.kernels import kernelLinear, kernelPoly, kernelRBF, kernelFourier
from Utilities.LFIRE_Estimator import PosteriorEstimator
from Utilities.ExpModels import  ArchModel

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model = ArchModel(-1,1, 0,1, randomSeed,T=1000,skip=10) 

#Number of samples to be created by each generator for classification.
n_samples=1000


#-------Load computed values from other scripts -------------------------------------------------------------------#
kRBF = lambda X1,X2: kernelRBF(X1,X2,1e-2)
clf=kC.KernelClassifier(kRBF)
regScale = np.linspace(1e-2,1e2,10)

kernelsAndRegScales = [(clf,regScale)]

# 2
with open('observedData.pickle', 'rb') as f:
     observedData = pickle.load(f)
(generatingParameters,obsData) = observedData


a = np.linspace(-0.9,0.9,10)
b = np.linspace(0.1,0.9,10)

x,y = np.meshgrid(a,b)
thetaVals = np.vstack([x.flatten(),y.flatten()]).T

thetaVals =thetaVals[60:80]

Estimator = PosteriorEstimator(randomSeed)
estimated_probabilities = Estimator.estimateProbabilities(obsData, thetaVals, model, n_samples, kernelsAndRegScales)
    
        

with open('estimated_probabilitiesH4.pickle', 'wb') as f:
    pickle.dump(estimated_probabilities, f)
    