import pickle
import numpy as np
from Utilities.ExpModels import  ArchModel

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#

with open('RBF_parametersAndRegScales.pickle', 'rb') as f:
     parametersAndRegScales = pickle.load(f)
[(T1,skip1,gamma1,regScale1)] = parametersAndRegScales

# Model
model = ArchModel(-1,1, 0,1, randomSeed,T1,skip1) 

#---Generate observed data ---------------------------------------#
#Number of datapoints to sample
n_datapoints = 100
observedData = model.generate_ObsData(n_datapoints)
with open('observedData.pickle', 'wb') as f:
    pickle.dump(observedData, f)