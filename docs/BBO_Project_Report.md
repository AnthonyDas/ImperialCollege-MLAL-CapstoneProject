# Black-Box Optimisation Project Report

## Initial Codebase

My pipeline was built from scratch using _pandas_, _numpy_, _scipy_, and _scikit-learn_ at its core. Part of my motivation for undertaking this course was to learn Python. Hence, writing a pipeline from scratch was a useful learning exercise. I also wanted the ability to see and tweak the “inner workings” of any model I used, and writing my own implementation supported that (e.g. when tuning Gaussian Process Regressor (GPR) hyperparameters, I switched between LML, MSE, SMSE and then my own SMSE variant as the selection criteria over time).

## Weekly Progress

**Weeks 1-3:**

Initial GP model codebase with limited kernel tuning between Matern, RBF, and Rational Quadratic based on their log marginal likelihood (LML). Used an Upper Confidence Bound (UCB) acquisition function with a single beta value per BBO Function.

**Week 4:**

When finding the UCB maximum, I started by evaluating it over an evenly spaced grid. Witnessing extremely small length scales for some input features, I realised that my acquisition function surface resolution was far too coarse; any rapid UCB surface changes occurring in between grid points would be missed entirely. I therefore switched to a two-stage strategy. I kept evaluating my current grid, but then selected the top 1000 x points as the starting seed points for a subsequent local optimisation (i.e. _scipy.optimize.minimize_).

**Week 5:**

Introduced kernel variants with an additive WhiteKernel to allow noise levels to be tuned rather than just relying on a fixed level within the GaussianProcessRegressor. I wanted to tune the noise level from the start of the BBO challenge, but didn’t understand how to until now. Also included MSE when predicting outputs for all x points. Notably, all x points were already part of the training dataset, so this was obvious data leakage. However, it mainly acted as a sanity check to see if the model was any good. E.g. I had difficulty modelling BBO Function 1 with its huge scale range. Even making predictions over x points in the training set led to wildly different predictions compared to the known outputs.

**Week 6.**

Increased my kernels’ length scale upper bound from 1 to 10. Initially, an upper bound of 1 was simply set because it represented the width of the x feature ranges. However, a higher value like 10 allowed the kernel optimiser to better “switch off” certain features it deemed irrelevant.

A significant improvement was switching my MSE calculation to be based on the Leave-One-Out Cross-Validation (LOOCV) score, so now it properly reflects predictions on unseen data. LOOCV MSE gave me a metric to really understand where my models’ predictive power was good or whether it was just overfitting to the training data.

**Week 7:**

To date, I had used transformers _sklearn.preprocessing_’s _StandardScaler_ and _QuantileTransformer_, and implemented my own arcsinh transformation. I wanted to encapsulate and apply the forward and backward transforms neatly via a _TransformedTargetRegressor_ wrapper. However, I failed to implement it due to TransformedTargetRegressor making the wrapped regressor inaccessible. This meant I could no longer retrieve GP member properties required by my pipeline. It was a good idea that unfortunately didn’t work out.

When an output transformation had been applied, I realised I wasn’t comparing like-for-like when using LOOCV MSE to compare different kernels. Technically, a fair comparison either required: 1.) models/kernels to fit over the same date, i.e. same output transformations, or 2.) LOOCV MSE to include a calculation back in the original space. Likewise, my UCB acquisition function calculations had become a mix of y-original space and y-transformed space. Thus, I applied inverse transforms where necessary to provide metrics in both y-transformed and y-original space.

Whilst the LOOCV MSE provided a means to compare different GP kernels (relative ability), I still didn’t have a good way to understand whether the models were any good (absolute ability). I finally found that the Standardised MSE (SMSE) could provide this. This was a major improvement as now I could identify which BBO Functions still had modelling issues and focus on them. Again, I calculated SMSE in both the y-transformed and the y-original space.

Introduced Expected Improvement (EI) acquisition function alongside UCB. Included more UCB beta and EI xi parameter values to run acquisition function optimisations for. This gave me a selection of possible next evaluation points for each BBO Function.

**Week 8:**

This week was mainly spent on pipeline quality of life improvements rather than actual BBO improvements. My Jupyter notebooks were updated to auto-reload imports (avoiding the pain of having to restart the notebook’s Python kernel each time an imported method changed), silenced user warnings to clean up my notebook’s output, and added try/except handling to prevent errors from stalling the pipeline.

**Week 9:**

With different kernels, x-transforms and y-transforms leading to a growing number of combinations to perform LOOCV over, my pipeline was taking too long to run. I found an analytical LOOCV formula which allowed the GP to fit the data only once when performing LOOCV (instead of needing to fit once for each and every data point). This provided a huge speed improvement and provided headroom for even more kernel choices to be trialled later on.

In BBO Function 5, due to my acquisition functions all suggesting the top right corner, which had already been evaluated, if scipy.optimize.minimize returned an already evaluated point, I stored and later reverse searched through its intermediate results to return the result of the latest search iteration that wasn’t an already evaluated point.   

**Week 10:**

Corrected a flaw in my inverse transformation from y-transformed back to y-original space. For non-linear transforms, the std must also be considered. Given that not all transforms have an analytical inversion formula, I resorted to Monte Carlo sampling to approximate the inverse transformation. I also wrote a dedicated class to provide a deterministic sequence of random normal samples that would be used by all Monte Carlo inversions, i.e. differences in random samples used wouldn’t be a factor in differences of Monte Carlo results.

**Week 11-12:**

Given these were the closing BBO weeks, I introduced a new metric to select between kernels when tuning, namely SMSE Top, which was the SMSE calculated over just the top quarter of outputs. This emphasised that I really only cared about the predictive accuracy over the highest outputs, given our BBO task was maximisation.

To aid the inclusion of more kernels and transforms, I standardised my transform and kernel class interfaces to allow more options to be trialled. This led me to further include transforms such as symmetric log and log shift transforms.

**Week 13:**

Included kernels such as ridge, local sensitivity and periodic kernels and combinations of these. In particular, with new “local sensitivity + ridge” and “locally periodic” kernels, I saw a major improvement in my best model’s SMSE for BBO Functions 1, 2 and 3, where I had always been struggling. These kernels finally saw my model start to approximate the underlying BBO function in a way that felt like it was really learning the output surface.

E.g. In BBO Function 2, my prior best GPR model was i.) _“Matern 0.5 + WhiteKernel, with no x transform, and logshift-scaled y-transform”_ which achieved a LOOCV SMSE of 0.50 in y-transformed space and 0.45 in y-original space. However, with ii.) _“locally periodic + WhiteKernel, with no x transform, and no y transform”_, the LOOCV SMSE plummeted to 0.04 in y-transformed space and 0.04 in y-original space.

## Final Reflections

* In weeks 12 and 13, I saw improvements in 4 of the BBO Functions. However, I attribute 2 of these to successful exploitation and 2 to the introduction of new kernel choices. In particular, I only finally started seeing real improvements in BBO Functions 1, 2 & 3 during the final few rounds of the BBO competition. My biggest regret was not including more kernel varieties earlier on, i.e. ridge, local sensitivity and periodic kernels. An important lesson has been how easy it is to generate new kernels using the base kernel set provided by _sklearn.gaussian_process_. Whilst I may not fully understand what the kernels actually represent, in my mind, it mirrors the tuning of hyperparameters in general. E.g. you don’t need to understand why a neural network performs best with N layers or M nodes; you just have to appreciate that it does. Similarly, I don’t need to fully understand what a particular kernel represents; I just need to find a kernel that models the data well, and that makes it useful. If I undertake BBO challenges or projects in future, I’ll remember to explore a wider array of kernels earlier on.

* I was surprised that my strategy and initial kernel options worked well in the higher-dimensional BBO Functions but less so in the lower-dimensional BBO Functions. Intuitively, the higher-dimensional spaces should pose more of a challenge, rather than the other way around. (Indeed, the BBO challenge rankings put 1st/51 participants in BBO Function 8 even though I had actually only submitted 12 queries at the time, with 9.997552331327 being my highest. My final query achieved an even higher value of 9.9999795682481). This reinforces the idea of trying more kernels earlier on. In particular, had I included a local sensitivity and locally periodic kernel earlier on, I would have progressed further in the lower-dimensional BBO Functions, but alas, I ran out of queries.

## Trade-offs and Decisions

* My pipeline is the same for all BBO Functions. It trials a GP model using different kernels and x and y transforms before selecting the best based on y-original space LOOCV SMSE. The biggest trade-off I faced was whether to spend my time researching new ideas targeting specific problematic BBO Functions (e.g. such as how to model the huge output range in BBO Function 1), or whether to implement general pipeline improvements that benefited all BBO Functions.

* This similarly applied when I considered whether to implement an entirely new type of model, e.g. a neural network or decision tree. In the end, I decided against branching out of a GP model because I always had a backlog of things I wanted to improve/fix in my GP pipeline, and I also didn’t think a vanilla neural network, say, was going to beat my already reasonably sophisticated GP model.

* My pipeline tests the entire set of kernels configured for every BBO Function. It does not remove kernels that performed poorly in earlier rounds, say. As such, excessive computational time initially prevented me from adding in more kernel variants. However, after I implemented analytical LOOCV, my notebooks all completed in 5-10 mins which was satisfactory. This enabled me to keep adding new kernel variants. LOOCV wasn’t a pure win, however. Given that I employ x and y transformations, the transformers themselves are fitted using all the data, including the point that is “left out”. Thus, there is slight data leakage, but the performance boost was worth this trade-off.

* Exploration and exploitation were balanced subjectively as I didn’t follow a pre-defined UCB beta/EI xi schedule, nor did I always enforce adjusting these parameter values downwards over successive BBO rounds. Instead, whilst I intended to become more exploitative over time, I typically reviewed the suggested next evaluation points from UCB and EI using a range of parameter values before selecting one based on achieving a high y-original space mean and being further away from any already evaluated x points. As I transitioned from exploration to exploitation, I similarly transitioned my focus from “highest distance to nearest x”, to “highest output mean”.

## Future Improvements

* I understand the distinction between exploration and exploitation. However, in the final BBO rounds, when I intended to be purely exploitative, I didn’t make any improvement in several BBO Functions. I wonder if there’s a heuristic to determine if exploitation has already been exhausted. Could any additional gain in fact just be noise, or indeed, what if the selected highest known output to exploit was itself a fluke caused by favourable noise? Perhaps I should have evaluated the exact same x point of the known high a second time to more reliably establish the noise level beforehand.

* My GP models for BBO Functions 3 and 6 are still able to predict +ve values even though their descriptions indicate they have a ceiling of zero. Further research is required to find a way to enforce the ceiling.

* I still haven’t found a satisfactory transform to handle the vast output scale differences in Function 1. I had attempted to implement beta warping, but it didn’t provide any improvement over my current best model. It also meant having to tune 2 parameters per input feature. This meant tuning 16 parameters for the 8 dimensions in BBO Function 8, as I wanted the same pipeline to be used for all BBO Functions. I should have seen this issue beforehand. Unfortunately, I spent a significant amount of time implementing beta warping, which I later dropped.

* I was uncomfortable with BBO Function 4, given the description said it was “full of local optima”. I relied on a distance measurement from the initial data’s high to my best submission as a rough guide as to whether I was on the same local optima. Instead, I should have been guided by plots of the GP surface to properly determine this. (Again, my desire to have the same pipeline for all BBO Functions was probably to blame for this, since a distance measurement is easy to apply everywhere, whereas plots needed to be tailored due to the varying dimensions across the BBO Functions).
