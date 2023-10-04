# Kernel Methods for Likelihood-Free Inference

## Description 
This project implements the [Kernel Logistic Regression (KLR) technique](https://hastie.su.domains/Papers/jcgs02_c.pdf) and investigates the application of KLR to the [LFIRE likelihood-free posterior-pdf estimation method](https://arxiv.org/pdf/1611.10242.pdf). This work was completed as part of my final-year undergraduate dissertation, under the guidance of Dr. Michael U. Gutmann at the University of Edinburgh.

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
```ClassifierEvaluation/kernelClassifier.py``` contians code that implements the KLR classifier.

```ClassifierEvaluation/classifierEvaluation_*``` notebooks tests the function of the KLR classifier against varied classification-problems. 

```ClassifierEvaluation/Gaussian_*_pdf_estimation.ipynb``` notebooks test the KLR classifier for toy-problem single-run gaussian pdf estimation using the LFIRE technique.

The common utility code for pdf-estimation experiments is seperated into the ```Utilities``` package (in the main directory).

**The Utilities package contains code for:**

* The kernel classifier
*  Kernel Functions
*  Data Generating models
*  Selection of hyper-parameters and regularization bands
*  LFIRE posterior estimation.

```Experiments``` contains jupyter-notebooks for experiments reported in **Dissertation.pdf**

**Each experiment in Experiments typicaly contains seperate scripts to:**

* Generate observed data points
* Select hyper-parameters and regularization bands
* Compute the posterior at multiple points (parallelized by having multiple scripts for different ranges of points)
* Coallate results and compute Evaluation metrics for the estimated posterior
* Visualization of experiments was conduted in IPython notebooks
