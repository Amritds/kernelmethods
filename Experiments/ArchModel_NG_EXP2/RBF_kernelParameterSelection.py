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
model = lambda T,skip: ArchModel(-1,1, 0,1, randomSeed,T=T,skip=skip) 
typicalThetaValue = np.array([0,0.5])
#Number of samples to be created by each generator for classification.
n_samples=1000

#----Select parameters -------------------------------------------#
#Select parameter values and reg scale for the polynomial kernel.
kRBF =lambda X1,X2: kernelRBF(X1,X2,1) #Any setting is fine.
clf=kC.KernelClassifier(kRBF)
(T,skip,gamma,regScale) = hC.autoARCH_GammaReal_and_RegScale(kernelRBF,clf,model,typicalThetaValue ,n_samples, numberOfOptions =20)

#----Save selections ---------------------------------------------#
parametersAndRegScales = [(T,skip,gamma,regScale)]

with open('RBF_parametersAndRegScales.pickle', 'wb') as f:
    pickle.dump(parametersAndRegScales, f)

