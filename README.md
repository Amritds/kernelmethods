# README #

## Description 
This project investigates the application of Kernel Logistic Regression for the LFIRE technique, completed for my Undergraduate 4th year project dissertation, under the supervision of Dr. Michael U. Gutmann at the University of Edinburgh.

The project dissertation report, **Dissertation.pdf** can be found in the main project directory, detailing work undertaken and results of the experiments conducted.

**Presentation.pdf** very briefly summarises the main results and conclusions.

## Dependencies ##
Python Version: **Python 3.6.3**

Package dependencies:

```numpy==1.23.2```
```scikit-learn==1.1.2```
```matplotlib==3.5.3```

Other dependencies can be installed by running ```conda install --name <env-name> --file spec-file.txt```

## Code ##
The ClassifierEvaluation directory code tests the KLR classifier implemented.

The ToyExamples directory contain very early experiments for posterior evaluation.

Experiments1 and Experiments2 directories contain experiments that informally investigated the research questions of the Dissertation.

Experiments for the presented results, using the Gaussian and ARCH(1) data generating models are largely contained within the Experiments3 directory.

The core code for a single experiment is common across most experiments and is seperated into the Utilities package (included in each experiment directory).

### The Utilities package contains code for:

The kernel classifier, 

Kernel Functions,

Data Generating models,

Selection of hyper-parameters and regularization bands,

LFIRE posterior estimation.

### Each experiment in Experiments3 typicaly has seperate scripts to:

Generate observed data points,

Select hyper-parameters and regularization bands,

Compute the posterior at multiple points (parallelized by having multiple scripts for different ranges of points),

Coallate results and compute Evaluation metrics for the estimated posterior.

Visualization of experiments was conduted in IPython notebooks.

The project reposotory also contains code for:

Experiments with the Fourier kernel from https://github.com/gmum/pykernels, which were not completed as part of the Dissertation. 

Experiments with the SSK kernel from the shogun machine learning library, using a Dynamic bistable HMM, which were not completed as part of the Dissertation. 
