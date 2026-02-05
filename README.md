# ImperialCollege-MLAL-CapstoneProject
Black Box Optimisation Capstone Project


## OVERVIEW

This Black-Box Optimisation (BBO) project replicates real-world engineering and machine learning scenarios where an unknown, multi-dimensional function needs to be optimised. Here, the optimisation goal is maximisation over eight different BBO functions with very limited starting data. Additionally, each BBO function may only be evaluated once per week, up to a maximum of 13 evaluations. These constraints replicate real-world scenarios where function evaluations are costly and/or time-consuming. E.g. the tuning of model hyperparameters, or devising optimal parameters for industrial manufacturing processes.  

Although I'm a career software developer in finance, I haven't had any direct exposure to applying machine learning (ML) or artificial intelligence (AI) techniques. As AI is fast becoming ubiquitous in all walks of life, completing a Professional Certificate in Machine Learning and Artificial Intelligence, of which this BBO project is a part, was my way to learn more about what exactly ML and AI are. The course also acted as a useful way to learn Python and its relevant modules.


## INPUTS AND OUTPUTS

* There are eight BBO functions in total, each varying between 2 and 8 input feature dimensions.
* All inputs are numeric spanning the unit range [0, 1).
* All functions have a single, continuous output variable which needs to be maximised. 
* Each function may only be queried once per week.
 

| Function   | Input | Output | Initial Samples | Optimisation |
|------------|-------|--------|-----------------|--------------|
| Function 1 | 2D    | 1D     | 10              | Maximisation |
| Function 2 | 2D    | 1D     | 10              | Maximisation |
| Function 3 | 3D    | 1D     | 15              | Maximisation |
| Function 4 | 4D    | 1D     | 30              | Maximisation |
| Function 5 | 4D    | 1D     | 20              | Maximisation |
| Function 6 | 5D    | 1D     | 20              | Maximisation |
| Function 7 | 6D    | 1D     | 30              | Maximisation |
| Function 8 | 8D    | 1D     | 40              | Maximisation |

For all functions, the input query format is x1 - x2 - x3 - ... - xn, where each xi starts with "0." and is specified to six decimal places. E.g. for Function 3, which has 3 feature dimensions, a valid input query would be: 0.444950-0.348788-0.558183. 

## CHALLENGE OBJECTIVE

As mentioned, the objective is to maximise each  BBO function. However, this isn't straightforward, bearing in mind:

* The sparse starting data vs the feature dimensionality.
* The limited available evaluations (13 in total per function).
* The six-decimal-place granularity for each feature dimension.
* Each BBO function mimics real-world complexity with hidden features such as non-linearity, multiple local maxima and noise.
* This BBO project runs alongside the latter half of the ML and AI Professional Certificate course. This means that new models and techniques are still being taught as the BBO project progresses. As such, direction and strategies are likely to evolve as our knowledge and capabilities expand.

## TECHNICAL APPROACH 

For my first three query submissions, I employed a Bayesian optimisation strategy and modelled each BBO function using a Gaussian Process (GP) surrogate with a Radial Basis Function (RBF) kernel. The selection of the next query submission point came from an Upper Confidence Bound (UCB) acquisition function where the chosen beta value reflected my desired balance between exploitation and exploration for the given BBO function. 

My overall strategy is to lean towards exploration in the earlier BBO rounds, before shifting towards exploitation in the final BBO rounds. I aim to uncover higher maxima or even the global maximum before locating its peak, rather than adopting exploitation now over what might only be a relatively low local optimum.
Additionally, I adopted more aggressive exploration for the lower-dimensional BBO functions, i.e. Functions 1 - 5. With only the initial data points and 13 available evaluations per BBO function, my intuition is that it isn't feasible to satisfactorily explore the 5D, 6D and 8D spaces, and hence the degree of exploration wasn't as aggressive.

The main challenge I encountered was setting the RBF kernel's lengthscale. I used an isotropic lengthscale, meaning all feature dimensions use the same lengthscale. Notably, from online reading, a good lower and upper bound for the lengthscale is the min and max distance between available data points, respectively. This was implemented, and the starting lengthscale was set to the midpoint between the lower and upper bounds. Additionally, to prevent getting stuck in a local optimum during hyperparameter tuning, a relatively high n_restarts_optimizer value of 20, and later 100, was used. (n_restarts_optimizer is "The number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood.") In later rounds, I aim to investigate using an anisotropic lengthscale such that each feature dimension has a bespoke lengthscale.  

## MODEL 

## HYPERPARAMETER OPTIMIZATION

## RESULTS

## CONTACT DETAILS
