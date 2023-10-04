import pickle
import numpy as np
import Utilities.kernelClassifier as kC
import Utilities.hyperparameterChoice as hC
from Utilities.ExpModels import  GaussianModel
from Utilities.kernels import kernelLinear, kernelPoly, kernelRBF

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model = GaussianModel(3,-20,20, randomSeed) 
typicalThetaValue = 0
#Number of samples to be created by each generator for classification.
n_samples=1000

#----Select parameters -------------------------------------------#
#Select parameter values and reg scale for the polynomial kernel.
kLin =lambda X1,X2: kernelLinear(X1,X2) #Any setting is fine.
clf=kC.KernelClassifier(kLin)
#-----------------------------------------------------------------#
allChoices =[]
mu_values = np.hstack([0,np.linspace(-20,20,10)])
for mu in mu_values:
    print('++++++++++++++++++++++++++ ',mu,' ++++++++++++++++++++++++++++++++++++')
    (regScale, scores) = hC.autoRegScale(clf,model,mu ,n_samples, numberOfOptions =20)
    choices = (regScale, scores)
    allChoices.append(choices)
#----Save selections ---------------------------------------------#
with open('choicesLINKernel.pickle', 'wb') as f:
    pickle.dump(allChoices, f)