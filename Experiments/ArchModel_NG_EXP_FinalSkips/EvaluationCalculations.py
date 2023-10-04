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
# 2
with open('estimated_probabilities.pickle', 'rb') as f:
     (theta_values,estimated_probabilities)  = pickle.load(f)

#--------Run Evaluation -----------------------------------------------------------------------#
(probabilities,classifiers_test_scores) = estimated_probabilities
probabilities = np.swapaxes(probabilities,0,1)

obj = ARCH_TP()
true_posterior = np.apply_along_axis(lambda obsData: obj.truePosterior(obsData*10,theta_values), 1, obsData)
np.save('TruePosterior',true_posterior)

kD1 = model.klDivergence(true_posterior,probabilities[0])
kD2 = model.klDivergence(true_posterior,probabilities[1])
kD3 = model.klDivergence(true_posterior,probabilities[2])
kD4 = model.klDivergence(true_posterior,probabilities[3])
np.save('RBF_klDivergence',kD1)
np.save('Standardized_RBF_klDivergence',kD2)
np.save('Fourier_klDivergence',kD3)
np.save('Standardized_Fourier_klDivergence',kD4)

