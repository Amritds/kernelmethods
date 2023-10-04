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
(probabilities,classifiers_test_scores) = estimated_probabilities
probabilities = np.swapaxes(probabilities,0,1)
kD1 = model.klDivergence(mu_values,obsData,probabilities[0])
kD2 = model.klDivergence(mu_values,obsData,probabilities[1])
kD3 = model.klDivergence(mu_values,obsData,probabilities[2])
kD4 = model.klDivergence(mu_values,obsData,probabilities[3])
np.save('POLY_klDivergence',kD1)
np.save('RBF_klDivergence',kD2)
np.save('Standardized_POLY_klDivergence',kD3)
np.save('Standardized_RBF_klDivergence',kD4)