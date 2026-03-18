# Model Card: Black-Box Optimisation Gaussian Process Model

## Overview

**Name:** Black‑Box Optimisation (BBO) Gaussian Process (GP) model.

The BBO is conducted for an educational capstone project run by Imperial College London towards the award of a Professional Certificate in Machine Learning and Artificial Intelligence.

**Type:** The model performs Bayesian Optimisation via a Gaussian Process (GP) surrogate employing both Upper Confidence Bound (UCB) and Expected Improvement (EI) acquisition functions to methodically select black-box optimisation evaluation points in a principled manner. The model has been coded in Python employing common libraries such as Numpy, Pandas, Scikit-learn and SciPy.

**Version:** 1.0

## Intended Use

### Suitable tasks

To optimise functions whose analytical form is unknown (i.e. “black-box” functions) that output a continuous, scalar value with and without the presence of noise. The model has been used to optimise functions whose inputs range from 2 \- 8 dimensions, with between 10 \- 53 data samples. The model is expected to work well up to 12 dimensions and hundreds of data samples. It may extend further, but it would need to be tested beforehand.

An example of a suitable task is hyperparameter tuning for a complex model where evaluations of the real model are monetarily costly and/or slow, either of which would lead to limited evaluations. As such, this BBO GP model can be used as a surrogate to efficiently guide hyperparameter tuning.

### Cases to Avoid

The model is not intended to optimise functions outputting categorical values without adaptation.

Whilst having more data samples in general is good, attempting to use a GP with thousands or tens of thousands of data points is likely to be prohibitively slow. This is because GPs involve matrix inversion, for which the computational complexity is O(N^3). Alternatives should be sought in this scenario.

It is not expected to be used in a commercial setting as it was developed for educational purposes only. Python source code has not been peer reviewed, and no warranty is provided as to its correctness.

## Details

The BBO task was maximisation of 8 independent BBO functions over the course of 13 available query submissions per BBO function. Notably, along with an initial dataset provided for each BBO function, successive query submissions had access to earlier submissions’ outputs. In this manner, the dataset to train the model expanded over time. This allowed the GP surrogate to be refined each time.   

### BBO Strategy

The overall strategy was to be more exploratory in the initial query submissions and then transition towards exploitation in the closing rounds. In this manner, the hope was to unveil higher output regions for each BBO function before exploiting the highest outputs. The rationale was to uncover the global peak’s region before performing exploitation, rather than performing exploitation too early over what could be just a local optimum. However, the balance between exploration and exploitation did not follow any set quantitative rule and was ultimately based on human judgment. The success of earlier query submissions to uncover higher outputs, or lack thereof, also played an influential role. 

### Techniques

As mentioned, a GP with UCB and EI acquisition functions is central to the model. Additionally, the model first trials multiple anisotropic kernels (e.g. Matern 1/2, Matern 3/2, Matern 5/2, RBF and Rational Quadratic), both with and without an additive White Kernel, and with and without a variety of x-input and y-output transforms, before selecting the best based on their leave-one-out cross-validation (LOOCV) Standardised Mean Squared Error (SMSE). Notably, the SMSE was initially calculated over all available y-outputs, but from the 11th submission query onwards, the SMSE was calculated using just the top 25% of y-outputs. Due to the BBO maximisation aim, kernel selection specifically focused on the predictive power over the highest outputs in the final BBO query submissions.  

## Performance

### Results Summary

After 11 of 13 available query submissions:

|   Function   | Initial Best |   BBO Best   | BBO Best Query   No. | Improvement   (%) |
|:------------:|:------------:|:------------:|:--------------------:|:-----------------:|
| Function   1 | 7.710875e-16 | 1.488979e-10 | 5                    | 19310017%         |
| Function   2 | 0.611205     | 0.632227     | 4                    | 3%                |
| Function   3 | -0.034835    | -0.007748    | 10                   | 78%               |
| Function   4 | -4.025542    |  0.702516    | 9                    | 117%              |
| Function   5 | 1088.86      | 8662.41      | 7                    | 696%              |
| Function   6 | -0.714265    | -0.146078    | 2                    | 80%               |
| Function   7 | 1.364968     | 2.785952     | 10                   | 104%              |
| Function   8 | 9.598482     | 9.997552     | 10                   | 4%                |

## Assumptions and Limitations

**Stationarity:** The GP surrogate model uses kernels (e.g. RBF, Matern) that assume the underlying BBO function is stationary. Both the smoothness and noise level ought to remain the same throughout the input space. If not, when the kernel optimiser tunes kernel parameters, values that might be appropriate for one input region might not perform well in other regions. Taking length scale as an example, if a given input feature is assumed to have a fixed length scale but in fact varies across the input space, as we attempt to climb an output peak, the model might take too large steps and overshoot the peak, or conversely, take too small steps and converge slowly.

**Kernel Parameter Bounds:** Currently, due to all 8 BBO functions having input features constrained to the range \[0, 1\] and submission queries being specified up to 6 d.p., White Kernel noise level bounds and kernel length scale bounds have been set to \[1e-7, 10\]. If a wider/tighter range is required, the Python code should be modified accordingly (see gaussian\_process\_helper.LOWER\_BOUND and gaussian\_process\_helper.UPPER\_BOUND).

### Failure modes

The GP model assumes outputs can assume both positive and negative values. Where underlying BBO functions have a known floor or ceiling, the model would need adaptation/extension to obey the floor/ceiling. 

## Ethical Considerations
