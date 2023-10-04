from __future__ import division
import time
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#-----------------------------------------DEFINE OBJECTIVE FUNCTION--------------------------------------------------------
#Sigmoid 
sigmoidNeg = lambda z: 1/(1+np.exp(-np.clip(z,-500,500)))
sigmoidPos = lambda z: 1/(1+np.exp(np.clip(z,-500,500)))
sigmoidInvNeg =lambda z: 1+np.exp(-z)
sigmoidInvPos =lambda z: 1+np.exp(z)

def careful_log_sigmoidInvNeg(z):
    if z>500:
        return 0
    elif z<-500:
        return -z
    else:
        return np.log(sigmoidInvNeg(z))
    
def careful_log_sigmoidInvPos(z):
    if z>500:
        return z
    elif z<-500:
        return 0
    else:
        return np.log(sigmoidInvPos(z))

#point loglikelihood---Assuming classes are 1 and -1 and no bias term
pLL=lambda w,ki,yi,bias:((1+yi)/2)*careful_log_sigmoidInvNeg(np.dot(w,ki)+bias)+((1-yi)/2)*(careful_log_sigmoidInvPos(np.dot(w,ki)+bias))

#Sum up pointwise log likelihood to get error function 
LogisticLoss= lambda w,K,y: (sum(pLL(w[1:],K[i],y[i],w[0]) for i in range(0,len(K))))/len(K)

#penalization term.
penalizationTerm =lambda w,K: np.dot(np.dot(w.T,K),w)/(len(K)*len(K))

#Add peanalization term and cost function-----(Kernelized)
penalizedKernelL2_func =lambda w,K,cost,lambda_coef: cost(w) + lambda_coef *penalizationTerm(w[1:],K)

#------------------------------DEFINE GRADIENT OF OBJECTIVE FUNCTION ------------------------------------------------------------

#pointwise gradient
pGrad=lambda w,ki,yi,bias: ((-(1+yi)/2)*sigmoidPos(np.dot(w,ki)+bias)+((1-yi)/2)*sigmoidNeg(np.dot(w,ki)+bias))*np.hstack([1,ki])

#Sum up pointwise gradient to get gradient of the cost function.
cost_Grad = lambda w,K,y: (sum(pGrad(w[1:],K[i],y[i],w[0]) for i in range(0,len(K))))/len(K)

#gradient of the penalization term.
penal_Grad= lambda w,K,y: 2*np.dot(K,w)/(len(K)*len(K))

#gradient of the penalizedKernelL2function
penalizedKernelL2_func_Grad= lambda w,K,y,lambda_coef: cost_Grad(w,K,y)+ lambda_coef*np.hstack([0,penal_Grad(w[1:],K,y)])

#--------------------------------------------Define kernelClassifier object-------------------------------------------------------

class KernelClassifier:
    def __init__(self,kernel, withStandardization=False, withScaling=True):
        self.kernel=kernel;
        self.flag=-1 #Set to 1 if model is trained.
        self.withScaling =withScaling
        self.withStandardization=withStandardization
    
    #Train a kernel classifier- logistic loss error function and L2 penalization.
    def train(self,X_train,y,lambda_coef=1,callback=None): 

        if(self.withStandardization):
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
        
        
        self.flag=1;
        self.X_train=X_train;
        self.pca=PCA()
        
        # K is the gram matrix
        K = self.kernel(X_train,X_train); 
        
        if(self.withScaling):
            #Scale the gram matrix.
            self.scale = np.mean(np.absolute(K))
            K=K/self.scale  
        
        #### Function handles - apply all parameters except the weight vector. (To be fitted by the minimizer) 
        funObj = lambda u:LogisticLoss(u,K,y);
        penalizedKernelL2 =lambda w: penalizedKernelL2_func(w,K,funObj,lambda_coef)
        self.penalizedKernelL2 = penalizedKernelL2
        
        Grad= lambda w: penalizedKernelL2_func_Grad(w,K,y,lambda_coef)
    
        # Now train the model.
        start_time = time.time()
        print('Training kernel logistic regression model...\n');
        self.weights = minimize(fun= penalizedKernelL2,x0 = np.zeros(1+X_train.shape[0]),method='L-BFGS-B',jac=Grad,callback=callback,options={'maxiter':100});
        print('COMPLETED. Running Time: %.3f seconds '% (time.time() - start_time));
        
    def predict(self,X):
        return (np.sign(self.activation(X)));
    
    def activation(self,X):
        if(self.flag==-1):
            print('ERROR!!! CLASSIFIER NOT TRAINED.')
            return 0;
        
        if(self.withStandardization):
            X = self.scaler.transform(X)
        
        
        K1=self.kernel(self.X_train,X).T
        
        if(self.withScaling):
            K1 =K1/self.scale #scale 
        
        w=self.weights.x 
        val=np.dot(K1,w[1:])+w[0] 
        
        return val;
    
    def prob(self,X):
        return sigmoidNeg(self.activation(X));
        
    def score(self,X,y):
        return 1-(np.count_nonzero((self.predict(X)*y.T)!=1)/len(y));
    
    def objectiveVal(self,weights):
        return self.penalizedKernelL2(weights)