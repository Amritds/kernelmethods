from __future__ import division
import numpy as np
import math
import matplotlib.mlab as mlab
import Utilities.kernelClassifier as kC
from sklearn import linear_model, model_selection
import matplotlib.pyplot as plt
import sys, os 
from scipy.stats import norm
from pathlib import Path


class PosteriorEstimator:
    def __init__(self,randomSeed = 432):
        #Random seed.
        np.random.seed(randomSeed)
        
    
    #Calculates desired ratio for observed data, selecting a kernel from a set of kernels.
    def calc_ratio(self, X_class1, X_class2, obsData,classifiersAndRegScales,n_samples,plotRegularization =False):

        #Setup the classification problem.
        y_vals_class1 = np.zeros((len(X_class1),1))+1
        y_vals_class2 = np.zeros((len(X_class2),1))-1

        y_vals = np.vstack([y_vals_class1,y_vals_class2])
        X_vals = np.vstack([X_class1,X_class2])

        datapoints=np.hstack([X_vals,y_vals])
        np.random.shuffle(datapoints)

        X_vals=datapoints[:,0:datapoints.shape[1]-1]
        y_vals=datapoints[:,datapoints.shape[1]-1]
    
        #Train, Test and Validation split. 
        X_vals_train = X_vals[:n_samples]
        y_vals_train = y_vals[:n_samples]

        X_vals_val = X_vals[n_samples:2*n_samples]
        y_vals_val = y_vals[n_samples:2*n_samples]
        
        X_vals_test = X_vals[2*n_samples:]
        y_vals_test = y_vals[2*n_samples:]
    
        #Compare kernels and choose the best classifier.
        bestScore = 0
        chosenKernel = -1
        log_ratio = 0
        
        classifiers_test_scores =np.zeros(len(classifiersAndRegScales))
        log_ratios = np.zeros((len(classifiersAndRegScales),len(obsData)))
        #For each kernel and corresponding regularization scale provided.
        for i in range(len(classifiersAndRegScales)):
            clf,regScale = classifiersAndRegScales[i]
            #Train the classifier and select regularization term.
            clf = self.classify(X_vals_train, X_vals_val, y_vals_train, y_vals_val, clf,  regScale, plotRegularization)
            
            #Get classification accuracy on a test set.
            score = clf.score(X_vals_test,y_vals_test)
            classifiers_test_scores[i] = score
            print("Classification accuracy on test set: ",score)
            if(bestScore < score):
                bestScore = score
                chosenKernel = i
            log_ratios[i] = clf.activation(obsData)
                
              
        print('Chosen classifier: ',chosenKernel)
        #Calculate ratio.
        ratios = np.exp(np.clip(log_ratios,-500,500))

        return (ratios,classifiers_test_scores)


    #Trains the classifier, choosing a regularizer on the validation set.
    def classify(self,X_vals_train, X_vals_val, y_vals_train, y_vals_val, clf, regScale,plotRegularization):
        bestScore = 0
        chosen_C= -1
        #Choose a regularization parameter.
        for c in regScale:
            print("Training for c value: ",c)
            clf.train(X_vals_train,y_vals_train,c)
            score = clf.score(X_vals_val,y_vals_val)
        
            if(plotRegularization):
                plt.plot(np.log(c)/np.log(10),score,'b*')
            if(chosen_C==-1 or bestScore < score):
                chosen_C= c
                bestScore = score
                
        clf.train(X_vals_train,y_vals_train,chosen_C)
        print("Chosen c value: ",chosen_C)
        return clf


    #input obsdata = n x single vector
    #return probabilities = n_obsData x parameter_values 
    def estimateProbabilities(self, observed_data, parameter_values, model, n_samples, classifiersAndRegScales):
        #Sample from the marginal.
        #Validation and Testing set equal to the size of the training set.
        
        file = Path('./marginalSamples.npy')
        if(file.is_file()):
            marginal_samples = np.load('marginalSamples.npy')
        else:
            marginal_samples = model.generator_marginal(3*n_samples)
            np.save('marginalSamples',marginal_samples)
        
        probabilities=np.zeros((len(parameter_values),len(classifiersAndRegScales),len(observed_data)))
        classifiers_test_scores = np.zeros((len(parameter_values),len(classifiersAndRegScales)))
        for i in range(len(parameter_values)):
            parameters = parameter_values[i]
            print("-----------CALCULATING parameter VALUES : ",parameters," -------------------")
            #Sample from the given class.
            #Validation and Testing set equal to the size of the training set.
            theta_samples=model.generator_paraGiven(parameters,3*n_samples)
    
            (ratios,scores) = self.calc_ratio(theta_samples,marginal_samples, observed_data, classifiersAndRegScales , n_samples)
            prob= model.prior*ratios
            probabilities[i] = prob
            classifiers_test_scores[i] = scores
        #Note probabilities is now: theta_value x classifier x observed data
        probabilities = np.swapaxes(probabilities,0,2)
        #Note probabilities is now: observed data x classifier x theta_value
        return (probabilities,classifiers_test_scores)