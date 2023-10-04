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
kPoly =lambda X1,X2: kernelPoly(X1,X2,1) #Any setting is fine.
clf=kC.KernelClassifier(kPoly)
#-----------------------------------------------------------------#
allChoices =[]
mu_values = np.linspace(-20,20,10)[2:4]
for mu in mu_values:
    print('++++++++++++++++++++++++++ ',mu,' ++++++++++++++++++++++++++++++++++++')
    (gamma,regScale, scores) = hC.autoGammaInt_and_RegScale(kernelPoly,clf,model,mu ,n_samples, numberOfOptions =20)
    choices = (gamma,regScale, scores)
    allChoices.append(choices)
#----Save selections ---------------------------------------------#
with open('choicesPOLYKernelQ2.pickle', 'wb') as f:
    pickle.dump(allChoices, f)
