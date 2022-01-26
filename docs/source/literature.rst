======
Papers
======

.. comment::
    
    Use the below template to add papers:

    Title
    --------------------------------------------------------------------
    - author, year :cite:p:`template_ref`
    - `PDF <url>`_
    
    **Abstract**
    
    
    *<abstract text>*
    
    **Notes**
    
    ...

Accounting for Variance in Machine Learning Benchmarks
------------------------------------------------------
- Bouthillier, 2021 :cite:p:`bouthillier2021`
- `PDF <https://proceedings.mlsys.org/paper/2021/file/cfecdb276f634854f3ef915e2e980c31-Paper.pdf>`_

**Abstract**

*Strong empirical evidence that one machine-learning algorithm A outperforms another one B, ideally calls for multiple trials optimizing the learning pipeline over sources of variation such as data sampling, augmentation, parameter initialization, and hyperparameters choices. This is prohibitively expensive, and corners are cut to reach conclusions. We model the whole benchmarking process and all sources of variation, revealing that variance due to data sampling, parameter initialization and hyperparameter choice impact markedly machine learning benchmark. We analyze the predominant comparison methods used today in the light of this variance. We show a counter-intuitive result that a biased estimator with more source of variation will give better results, closer to the ideal estimator at a 51x reduction in compute cost. Using this we perform a detailed study on the error rate of detecting improvements, on five different deep-learning tasks/architectures. This study leads us to propose recommendations for future performance comparisons.*

**Notes**

...

A Bayesian interpretation of the confusion matrix
-------------------------------------------------
- Caelen, 2017 :cite:p:`caelen2017`
- `PDF <http://www.oliviercaelen.be/doc/confMatrixBayes_AMAI.pdf>`_

**Abstract**

*We propose a way to infer distributions of any performance indicator computed from the confusion matrix. This allows us to evaluate the variability of an indicator and to assess the importance of an observed difference between two performance indicators. We will assume that the values in a confusion matrix are observations coming from a multinomial distribution. Our method is based on a Bayesian approach in which the unknown parameters of the multinomial proba- bility function themselves are assumed to be generated from a random vector. We will show that these unknown parameters follow a Dirichlet distribution. Thanks to the Bayesian approach, we also benefit from an elegant way of injecting prior knowledge into the distributions. Experiments are done on real and synthetic data sets and assess our method's ability to construct accurate distributions.*

**Notes**

...

Cross-validation: what does it estimate and how well does it do it?
-------------------------------------------------------------------
- Bates, 2021 :cite:p:`bates2021`
- `PDF <https://arxiv.org/pdf/2104.00673>`_

**Abstract**

*Cross-validation is a widely-used technique to estimate prediction error, but its behavior is complex and not fully understood. Ideally, one would like to think that cross-validation estimates the prediction error for the model at hand, fit to the training data. We prove that this is not the case for the linear model fit by ordinary least squares; rather it estimates the average prediction error of models fit on other unseen training sets drawn from the same population. We further show that this phenomenon occurs for most popular estimates of prediction error, including data splitting, bootstrapping, and Mallow's Cp. Next, the standard confidence intervals for prediction error derived from cross-validation may have coverage far below the desired level. Because each data point is used for both training and testing, there are correlations among the measured accuracies for each fold, and so the usual estimate of variance is too small. We introduce a nested cross-validation scheme to estimate this variance more accurately, and show empirically that this modification leads to intervals with approximately correct coverage in many examples where traditional cross-validation intervals fail. Lastly, our analysis also shows that when producing confidence intervals for prediction accuracy with simple data splitting, one should not re-fit the model on the combined data, since this invalidates the confidence intervals.*

**Notes**

...

Classifier uncertainty: evidence, potential impact, and probabilistic treatment
-------------------------------------------------------------------------------
- Totsch, 2021 :cite:p:`totsch2021`
- `PDF <https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC7959610&blobtype=pdf>`_

**Abstract**

*Cross-validation is a widely-used technique to estimate prediction error, but its behavior is complex and not fully understood. Ideally, one would like to think that cross-validation estimates the prediction error for the model at hand, fit to the training data. We prove that this is not the case for the linear model fit by ordinary least squares; rather it estimates the average prediction error of models fit on other unseen training sets drawn from the same population. We further show that this phenomenon occurs for most popular estimates of prediction error, including data splitting, bootstrapping, and Mallow's Cp. Next, the standard confidence intervals for prediction error derived from cross-validation may have coverage far below the desired level. Because each data point is used for both training and testing, there are correlations among the measured accuracies for each fold, and so the usual estimate of variance is too small. We introduce a nested cross-validation scheme to estimate this variance more accurately, and show empirically that this modification leads to intervals with approximately correct coverage in many examples where traditional cross-validation intervals fail. Lastly, our analysis also shows that when producing confidence intervals for prediction accuracy with simple data splitting, one should not re-fit the model on the combined data, since this invalidates the confidence intervals.*

**Notes**

...


============
Bibliography
============

.. bibliography::
   :style: plain
