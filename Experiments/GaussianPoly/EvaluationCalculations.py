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
with open('estimated_probabilitiesH1.pickle', 'rb') as f:
     (mu_values,estimated_probabilities)  = pickle.load(f)

(probabilities,classifiers_test_scores) = estimated_probabilities
probabilities = np.swapaxes(probabilities,0,1)
print(probabilities.shape,' ',mu_values.shape)
p1 = probabilities[1]
p2 = probabilities[2]
muVals = mu_values
with open('estimated_probabilitiesH2.pickle', 'rb') as f:
     (mu_values,estimated_probabilities)  = pickle.load(f)

(probabilities,classifiers_test_scores) = estimated_probabilities
probabilities = np.swapaxes(probabilities,0,1)
p1 = np.hstack([p1,probabilities[1]])
p2 = np.hstack([p2,probabilities[3]])
muVals = np.hstack([muVals,mu_values])
with open('estimated_probabilitiesH3.pickle', 'rb') as f:
     (mu_values,estimated_probabilities)  = pickle.load(f)

(probabilities,classifiers_test_scores) = estimated_probabilities
probabilities = np.swapaxes(probabilities,0,1)
p1 = np.hstack([p1,probabilities[1]])
p2 = np.hstack([p2,probabilities[3]])
muVals = np.hstack([muVals,mu_values])
with open('estimated_probabilitiesH4.pickle', 'rb') as f:
     (mu_values,estimated_probabilities)  = pickle.load(f)

(probabilities,classifiers_test_scores) = estimated_probabilities
probabilities = np.swapaxes(probabilities,0,1)
p1 = np.hstack([p1,probabilities[1]])
p2 = np.hstack([p2,probabilities[3]])
muVals = np.hstack([muVals,mu_values])
with open('estimated_probabilitiesH5.pickle', 'rb') as f:
     (mu_values,estimated_probabilities)  = pickle.load(f)

(probabilities,classifiers_test_scores) = estimated_probabilities
probabilities = np.swapaxes(probabilities,0,1)
p1 = np.hstack([p1,probabilities[1]])
p2 = np.hstack([p2,probabilities[3]])
muVals = np.hstack([muVals,mu_values])


#--------Run Evaluation -----------------------------------------------------------------------#
print(muVals.shape,' ',obsData.shape,' ',p1.shape)
obsData = np.vstack([0,obsData])
kD2 = model.klDivergence(muVals,obsData,p1)
kD4 = model.klDivergence(muVals,obsData,p2)
np.save('KL_POLY_degree2',kD2)
np.save('KL_POLY_degree4',kD4)

np.save('ObsPolyprob_deg2',p1)
np.save('ObsPolyprob_deg4',p2)
        
