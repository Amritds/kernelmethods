import numpy as np
import pickle
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
# 1
(generatingParameters,obsData) = np.load('ObservedData.npy')
# 2
with open('estimated_probabilities.pickle', 'rb') as f:
     (mu_values,estimated_probabilities)  = pickle.load(f)

#--------Run Evaluation -----------------------------------------------------------------------#
kD = model.klDivergence(mu_values,obsData,estimated_probabilities)
np.save('klDivergence',kD)