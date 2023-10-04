import pickle
import numpy as np
import Utilities.kernelClassifier as kC
from Utilities.kernels import kernelLinear, kernelPoly, kernelRBF
from Utilities.LFIRE_Estimator import PosteriorEstimator
from Utilities.ExpModels import relativeError
from Utilities.ExpModels import  ArchModel

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
# 2
with open('estimated_probabilities.pickle', 'rb') as f:
     (mu_values,estimated_probabilities)  = pickle.load(f)

#--------Run Evaluation -----------------------------------------------------------------------#
(probabilities,classifiers_test_scores) = estimated_probabilities
probabilities = np.swapaxes(probabilities,0,1)
kD1 = model.klDivergence(mu_values,obsData,probabilities[0])
kD2 = model.klDivergence(mu_values,obsData,probabilities[1])
np.save('POLY_klDivergence',kD1)
np.save('RBF_klDivergence',kD2)