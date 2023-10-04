import pickle
import numpy as np
import Utilities.kernelClassifier as kC
import Utilities.hyperparameterChoice as hC
from Utilities.ExpModels import  ArchModel
from Utilities.kernels import kernelLinear, kernelPoly, kernelRBF, kernelFourier

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model1 = ArchModel(-1,1, 0,1, randomSeed) 

# Model
model2 = ArchModel(-1,1, 0,1, randomSeed,T=1000,skip=10) 
#Number of samples to be created by each generator for classification.
n_samples=1000

#----Select parameters -------------------------------------------#
#Select parameter values and reg scale for the polynomial kernel.
kRBF =lambda X1,X2: kernelRBF(X1,X2,1) #Any setting is fine.
clf1=kC.KernelClassifier(kRBF)

kF =lambda X1,X2: kernelFourier(X1,X2,1) #Any setting is fine.
clf2=kC.KernelClassifier(kF)
#-----------------------------------------------------------------#

expno = 1

allChoices1 =[]
allChoices2 =[]
allChoices3 =[]
allChoices4 =[]


a = np.linspace(-1,1,5)
b = np.linspace(0,0.5,5)

x,y = np.meshgrid(a,b)
thetaVals = np.vstack([x.flatten(),y.flatten()]).T

for thetaval in thetaVals[0:5]:
 #   print('++++++++++++++++++++++++++ ',thetaval,' ++++++++++++++++++++++++++++++++++++')
  #  (gamma,regScale, scores) = hC.autoGammaReal_and_RegScale(kernelRBF,clf1,model1,thetaval ,n_samples, numberOfOptions =20)
   # choices = (gamma,regScale, scores)
   # allChoices1.append(choices)
    
    print('++++++++++++++++++++++++++ ',thetaval,' ++++++++++++++++++++++++++++++++++++')
    (gamma,regScale, scores) = hC.autoGammaReal_and_RegScale(kernelFourier,clf2,model1,thetaval ,n_samples, numberOfOptions =20)
    choices = (gamma,regScale, scores)
    allChoices2.append(choices)
    
   # print('++++++++++++++++++++++++++ ',thetaval,' ++++++++++++++++++++++++++++++++++++')
   # (gamma,regScale, scores) = hC.autoGammaReal_and_RegScale(kernelRBF,clf1,model2,thetaval ,n_samples, numberOfOptions =20)
   # choices = (gamma,regScale, scores)
   # allChoices3.append(choices)
    
   # print('++++++++++++++++++++++++++ ',thetaval,' ++++++++++++++++++++++++++++++++++++')
   # (gamma,regScale, scores) = hC.autoGammaReal_and_RegScale(kernelRBF,clf2,model2,thetaval ,n_samples, numberOfOptions =20)
   # choices = (gamma,regScale, scores)
   # allChoices4.append(choices)
    #----Save selections ---------------------------------------------#
   # with open('choicesRBFKernelQ' + str(expno) + '.pickle', 'wb') as f:
    #    pickle.dump(allChoices1, f)
    
    with open('choicesFourierKernelQ' + str(expno) + '.pickle', 'wb') as f:
        pickle.dump(allChoices2, f)
    
  #  with open('skip_choicesRBFKernelQ' + str(expno) + '.pickle', 'wb') as f:
   #     pickle.dump(allChoices3, f)

   # with open('skip_choicesFourierKernelQ' + str(expno) + '.pickle', 'wb') as f:
    #    pickle.dump(allChoices4, f)
        
with open('doneQ' + str(expno) + '.pickle', 'wb') as f:
    pickle.dump(4, f)

       