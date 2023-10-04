import pickle
import numpy as np
import Utilities.kernelClassifier as kC
from Utilities.kernels import kernelLinear, kernelPoly, kernelRBF
from Utilities.LFIRE_Estimator import PosteriorEstimator
from Utilities.ExpModels import relativeError
from Utilities.ExpModels import  ArchModel, ARCH_TP

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model = ArchModel(-1,1, 0,1, randomSeed) 
#Number of samples to be created by each generator for classification.
n_samples=1000

#-------Load computed values from other scripts -------------------------------------------------------------------#
# 1
with open('observedData.pickle', 'rb') as f:
     (generatingParameters,obsData) = pickle.load(f)

#--------Run Evaluation -----------------------------------------------------------------------#
a = np.linspace(-0.9,0.9,10)
b = np.linspace(0.1,0.9,10)

x,y = np.meshgrid(a,b)
thetaVals = np.vstack([x.flatten(),y.flatten()]).T

obj = ARCH_TP()
true_posterior = np.apply_along_axis(lambda obsData: obj.truePosterior(obsData*10,thetaVals), 1, obsData)
np.save('TruePosterior',true_posterior)

