import numpy as np
from Utilities.ExpModels import  GaussianModel

#Random Seed
randomSeed = 432

#------Exp Setup -------------------------------------------------#
# Model
model = GaussianModel(3,-2,2, randomSeed) 

#---Generate observed data ---------------------------------------#
#Number of datapoints to sample
n_datapoints = 100
(generatingParameters,obsData) = model.generate_ObsData(n_datapoints)
np.save('ObservedData',(generatingParameters,obsData))
