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

#----Select parameters -------------------------------------------#
#Select parameter values and reg scale for the polynomial kernel.
kRBF =lambda X1,X2: kernelRBF(X1,X2,1) #Any setting is fine.
clf=kC.KernelClassifier(kRBF)
#-----------------------------------------------------------------#

(X_test,y_test) = hC.returnSetup(0,model,2500)


allChoices =[]
size_values = np.array([100,200,500,1000,1500,2000,2500])
for n_samples in size_values:
    print('++++++++++++++++++++++++++ ',n_samples,' ++++++++++++++++++++++++++++++++++++')
    (gamma,reg,X_train,y_train) = hC.autoGammaReal_and_RegScaleBestScore(kernelRBF,clf,model,typicalThetaValue ,n_samples, numberOfOptions =20)
    
    kRBF =lambda X1,X2: kernelRBF(X1,X2,gamma) 
    clf=kC.KernelClassifier(kRBF)
    clf.train(X_train,y_train,reg)
    testScore = clf.score(X_test,y_test)
    
    choices = (gamma,reg,testScore)
    allChoices.append(choices)
    #----Save selections ---------------------------------------------#
    with open('numbersScores.pickle', 'wb') as f:
        pickle.dump(allChoices, f)