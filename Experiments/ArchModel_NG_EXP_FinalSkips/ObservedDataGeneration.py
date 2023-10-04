import pickle
import numpy as np
from Utilities.ExpModels import  ArchModel

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model = ArchModel(-0.25,0.25, 0.375,0.625, randomSeed,T=1000,skip=10) 

#---Generate observed data ---------------------------------------#
#Number of datapoints to sample
n_datapoints = 100
(y_t,observedData) = model.generate_ObsData(n_datapoints)

with open('y_t.pickle', 'wb') as f:
    pickle.dump(y_t, f)

with open('observedData.pickle', 'wb') as f:
    pickle.dump(observedData, f)