import pickle
import numpy as np
from Utilities.ExpModels import  ArchModel

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model = ArchModel(-1,1, 0,1, randomSeed) 

#---Generate observed data ---------------------------------------#
#Number of datapoints to sample
n_datapoints = 100
observedData = model.generate_ObsData(n_datapoints)
with open('observedData.pickle', 'wb') as f:
    pickle.dump(observedData, f)