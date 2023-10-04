from sklearn import model_selection
import numpy as np

#################################################### Helper functions ###############################################################

def classify(X_vals,y_vals,clf,scale):
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X_vals,y_vals,test_size=0.30)
    
    scores = np.zeros(len(scale))
    bestScore = 0
    chosen_C= -1

    kf = model_selection.KFold(n_splits=5)
    #Choose a regularization parameter.
    for i in range(len(scale)):
        c =scale[i]
        print("Training for c value: ",c)
        sum = 0
        for train, test in kf.split(X_vals):
            clf.train(X_vals[train],y_vals[train],c)
            sum += clf.score(X_vals[test],y_vals[test])
        average = sum/5
        scores[i] =average
    return scores


def returnSetup(theta_values, model, n_samples):
    #Sample from the marginal.
    marginal_samples=model.generator_marginal(n_samples)

    #Sample from the given class.
    theta_samples=model.generator_paraGiven(theta_values,n_samples)
    
    #-------------------------------------- Setup the classification task.
    X_class1 = theta_samples
    X_class2= marginal_samples
        
    y_vals_class1 = np.zeros((len(X_class1),1))+1
    y_vals_class2 = np.zeros((len(X_class2),1))-1

    y_vals = np.vstack([y_vals_class1,y_vals_class2])
    X_vals = np.vstack([X_class1,X_class2])

    datapoints=np.hstack([X_vals,y_vals])
    np.random.shuffle(datapoints)

    X_vals=datapoints[:,0:datapoints.shape[1]-1]
    y_vals=datapoints[:,datapoints.shape[1]-1]
    #----------------------------------------------------------------------    
    return (X_vals,y_vals)

def returnMaxRange(scores,scale,percentage):
    #Start with heavy regularization and reduce until a good score for the task is approached. Choose this value as the minRange.
    diff = np.max(scores) - scores[len(scores)-1]
    maxRange =-1 
    for i in range(len(scores)):
        j= len(scores) -1 -i
        if scores[j] - scores[len(scores)-1] > (percentage*diff):
            maxRange = scale[j]
            break;
    if(maxRange == -1):
        print("WARNING WARNING CHOOOSE A LARGER REGULARIZATION SCALE")
    
    return maxRange

def returnChosenVal(scores,scaleGamma,percentage):
    #No need to be too exact in this selection as errors are corrected by regularizaion term.
    #start with small gamma parameter and increase until a good score for the task is approached. Choose this as the gamma parameter.
    diff = np.max(scores) - np.max(scores[0])
    chosenGamma = -1
    for i in range(len(scaleGamma)):
        if np.max(scores[i]) - np.max(scores[0]) > (percentage*diff):
            chosenGamma = i
            break;
    if(chosenGamma == -1):
        print("WARNING WARNING CHOOOSE A SMALLER GAMMA SCALE")
        chosenGamma = 0
    return chosenGamma


################################## Call these functions to make hyperparameter choices. ##############################################

# -------------------------------- For selection of  1. kernel regularization scale -----------------------------------------------
def autoRegScale(clf,model,theta_values,n_samples,numberOfOptions):
    #----------------------------------- Tunining Parameters
    # Regualrization scale.
    width  = 1e-4
    percentage = 0.90
    scale = np.array([1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12])
    
    #----------------------------------- Get Selection
    # Get  classification setup
    (X_vals,y_vals) = returnSetup(theta_values, model, n_samples)

    # Get cross validated scores for different parameter settings.
    scores = classify(X_vals,y_vals,clf,scale)
    
    # Make selections.
    maxRange = returnMaxRange(scores,scale,percentage)
    
    # Return a good regularization scale.
    return np.linspace(width*maxRange,maxRange,numberOfOptions)
    
# ------------------------------- For selection of 1. kernel real value parameter gamma,
#                                                  2. kernel regularization scale. --------------------------------------------------
def autoGammaReal_and_RegScale(kernelRAW,clf,model,theta_values,n_samples,numberOfOptions):
    #--------------------------------- Tunining Parameters
    width =1e-4
    percentage  =0.90
    scale = np.array([1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12])
    
    scaleGamma =scale[3:15] # Not exploring overly complicated models
    
    #-------------------------------- Get Selection
    # Get  classification setup
    (X_vals,y_vals) = returnSetup(theta_values, model, n_samples)

    # Get cross validated scores for different parameter settings.
    scores =np.zeros((len(scaleGamma),len(scale)))
    for i in range(len(scaleGamma)):              
        print('############### Calculating for gamma value = ',scaleGamma[i],' #####################')
        clf.kernel = lambda X1,X2: kernelRAW(X1,X2,scaleGamma[i])
        scores[i] = classify(X_vals,y_vals,clf,scale)
    
    # Make selections
    chosenGamma = returnChosenVal(scores,scaleGamma,percentage)
    maxRange = returnMaxRange(scores[chosenGamma],scale,percentage)
            
    # Return a good regularization scale and choice of gamma.
    return (scaleGamma[chosenGamma],np.linspace(width*maxRange,maxRange,numberOfOptions),scores)

# ------------------------------- For selection of 1. kernel integer value parameter gamma,
#                                                  2. kernel regularization scale. --------------------------------------------------
def autoGammaInt_and_RegScale(kernelRAW,clf,model,theta_values,n_samples,numberOfOptions):
    #--------------------------------- Tunining Parameters
    width =1e-4
    percentage  =0.90
    scale = np.array([1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12])

    scaleGamma =np.array([1,2,3,4,5]) # Not exploring overly complicated models
    
    #-------------------------------- Get Selection
    # Get  classification setup
    (X_vals,y_vals) = returnSetup(theta_values, model, n_samples)

    # Get cross validated scores for different parameter settings.
    scores =np.zeros((len(scaleGamma),len(scale)))
    for i in range(len(scaleGamma)):              
        print('############### Calculating for gamma value = ',scaleGamma[i],' #####################')
        clf.kernel = lambda X1,X2: kernelRAW(X1,X2,scaleGamma[i])
        scores[i] = classify(X_vals,y_vals,clf,scale)
    
    # Make selections
    chosenGamma = returnChosenVal(scores,scaleGamma,percentage)
    maxRange = returnMaxRange(scores[chosenGamma],scale,percentage)
            
    # Return a good regularization scale and choice of gamma.
    return (scaleGamma[chosenGamma],np.linspace(width*maxRange,maxRange,numberOfOptions),scores)


# ------------------------------- For selection of 1. kernel integer value parameter beta,
#                                                  3. kernel real value gamma less than 1
#                                                  2. kernel regularization scale. --------------------------------------------------
def autoBetaInt_GammaReal_and_RegScale(kernelRAW,clf,model,theta_values,n_samples,numberOfOptions):
    #--------------------------------- Tunining Parameters
    width =1e-4
    percentage  =0.90
    scale = np.array([1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12])
    
    scaleBeta  =np.array([1,2,3,4,5,6,7,8,9,10]) # Not exploring overly complicated models
    scaleGamma =scale[:12] 
    
    #-------------------------------- Get Selection
    # Get  classification setup
    (X_vals,y_vals) = returnSetup(theta_values, model, n_samples)

    # Get cross validated scores for different parameter settings.
    scores =np.zeros((len(scaleBeta),len(scaleGamma),len(scale)))
    for i in range(len(scaleBeta)): 
        for j in range(len(scaleGamma)):
            print('############### Calculating for beta value = ',scaleBeta[i],' , gamma value = ',scaleGamma[j],' #####################')
        clf.kernel = lambda X1,X2: kernelRAW(X1,X2,scaleBeta[i],scaleGamma[j])
        scores[i,j] = classify(X_vals,y_vals,clf,scale)
    
    # Make selections
    chosenBeta = returnChosenVal(scores,scaleBeta,percentage)
    chosenGamma = returnChosenVal(scores[chosenBeta],scaleGamma,percentage)
    maxRange = returnMaxRange(scores[chosenGamma],scale,percentage)
            
    # Return a good regularization scale and choices of beta,gamma.
    return (scaleBeta[chosenBeta],scaleGamma[chosenGamma],np.linspace(width*maxRange,maxRange,numberOfOptions))

def autoGammaReal_and_RegScaleBestScore(kernelRAW,clf,model,theta_values,n_samples,numberOfOptions):
    #--------------------------------- Tunining Parameters
    width =1e-4
    percentage  =0.90
    scale = np.array([1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12])
    
    scaleGamma =scale[:12] # Not exploring overly complicated models
    
    #-------------------------------- Get Selection
    # Get  classification setup
    (X_vals,y_vals) = returnSetup(theta_values, model, n_samples)

    # Get cross validated scores for different parameter settings.
    scores =np.zeros((len(scaleGamma),len(scale)))
    for i in range(len(scaleGamma)):              
        print('############### Calculating for gamma value = ',scaleGamma[i],' #####################')
        clf.kernel = lambda X1,X2: kernelRAW(X1,X2,scaleGamma[i])
        scores[i] = classify(X_vals,y_vals,clf,scale)
    
    # Make selections
    chosenGamma = returnChosenVal(scores,scaleGamma,percentage)
    maxRange = returnMaxRange(scores[chosenGamma],scale,percentage)
    
    # Return a good regularization scale and choice of gamma.
    return (scaleGamma[chosenGamma],scale[np.argmax(scores[chosenGamma])],X_vals,y_vals)
