import numpy as np
from shogun import SubsequenceStringKernel, StringCharFeatures, RAWBYTE
#------------------------------------------DEFINE KERNELS-------------------------------------------------------------------------
#Linear kernel.
def kernelLinear(X1,X2):
    return np.dot(X1,X2.T) #Return the gram matrix with inner product as standard dot product.

#Polynomial kernel.
def kernelPoly(X1,X2,d):
    return np.power((kernelLinear(X1,X2)+1),d) #Return the gram matrix with inner product as standard dot product.

#RBF kernel.
#def kernelRBF(X1,X2,sigma):
 #   Z = 1/np.sqrt(2*3.1416*np.power(sigma,2));
  #  D= np.dot(np.power(X1,2),np.ones((X1.shape[1],X2.shape[0]))) + np.dot(np.ones((X1.shape[0],X1.shape[1])),np.power(X2,2).T) - 2*np.dot(X1,X2.T);
#    return Z*np.exp(-D/(2*np.power(sigma,2)));
#RBF kernel
def kernelRBF(X1,X2,gamma):
    D= np.dot(np.power(X1,2),np.ones((X1.shape[1],X2.shape[0]))) + np.dot(np.ones((X1.shape[0],X1.shape[1])),np.power(X2,2).T) - 2*np.dot(X1,X2.T);
    return np.exp(-gamma*D);

def stringKernel(X1,X2,k_coeff,lambda_coeff):
    # X1, X2 must be lists of strings
    if type(X1) == np.ndarray:
        X1 = X1.tolist()
    if type(X2) == np.ndarray:
        X2 = X2.tolist()
    
    #Create shogun objects
    features1 = StringCharFeatures(X1, RAWBYTE)
    features2 = StringCharFeatures(X2, RAWBYTE)
    
    #Use shogun string kernel.
    # k and lambda are as described in Lodhi 2002
    sk = SubsequenceStringKernel(features1, features2, k_coeff, lambda_coeff)
    
    return sk.get_kernel_matrix()
    