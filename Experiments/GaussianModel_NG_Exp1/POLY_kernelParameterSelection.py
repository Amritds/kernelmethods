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
#Select parameter values and reg scale for the rbf kernel.
kPoly =lambda X1,X2: kernelPoly(X1,X2,1) #Any setting is fine.
clf=kC.KernelClassifier(kPoly)
(gamma,regScale) = hC.autoGammaInt_and_RegScale(kernelPoly,clf,model,typicalThetaValue ,n_samples, numberOfOptions =20)

#----Save selections ---------------------------------------------#
parametersAndRegScales =[(gamma,regScale)]

with open('POLY_parametersAndRegScales.pickle', 'wb') as f:
    pickle.dump(parametersAndRegScales, f)