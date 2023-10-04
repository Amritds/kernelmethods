import pickle
import numpy as np
import Utilities.kernelClassifier as kC
import Utilities.hyperparameterChoice as hC
from Utilities.ExpModels import  ArchModel
from Utilities.kernels import kernelLinear, kernelPoly, kernelRBF

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model = ArchModel(-1,1, 0,1, randomSeed) 
typicalThetaValue = np.array([0,0.5])
#Number of samples to be created by each generator for classification.
n_samples=1000

#----Select parameters -------------------------------------------#
#Select parameter values and reg scale for the polynomial kernel.
kRBF =lambda X1,X2: kernelRBF(X1,X2,1) #Any setting is fine.
clf=kC.KernelClassifier(kRBF)
(gamma,regScale) = hC.autoGammaReal_and_RegScale(kernelRBF,clf,model,typicalThetaValue ,n_samples, numberOfOptions =20)

#----Save selections ---------------------------------------------#
parametersAndRegScales = [(gamma,regScale)]

with open('RBF_parametersAndRegScales.pickle', 'wb') as f:
    pickle.dump(parametersAndRegScales, f)

