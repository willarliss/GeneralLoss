# `GeneralLossMinimizer`
```
minimizers.minimize.GeneralLossMinimizer(loss_fn=None, link_fn=None, penalty='none', alpha=0.1, l1_ratio=0.15, solver='bfgs', tol=1e-4, max_iter=1000, verbose=0, fit_intercept=True, random_state=None, options=None)
```
### Description
pass
### Parameters
 - loss_fn: Callable, default=None
   - Loss function to optimize during training. Should accept a Nx1 array of targets and an Nx1 array of predictions. Should return a Nx1 array of losses. If not specified, defaults to binary cross-entropy for _estimator_type='classifier' and mean squared error for _estimator_type='regressor'.
 - link_fn: Callable, default=None
   - Link function to make predictions. Should accept a NxP array of observation data and a 1xP array of parameters. Should return an Nx1 array of predictions. If not specified, defaults to the sigmoid function for _estimator_type='classifier' and a linear combination for _estimator_type='regressor'.
 - penalty: Callable or str, default='none'
   - Penalthy function to regularize parameters during training. Can be 'none' for no penalty, 'l1' for a L1-penalty, 'l2' for a L2-penalty, 'elasticnet' for an elastic-net penalty, or a callable function. If a function, it should accept a 1xP array of parameters and return a single float. Defaults to no penalty.
 - alpha: float, default=0.1
   - Penalty strength parameter. Constant by which to multiply the penalty function. Should be greater than 0.
 - l1_ratio: float, default=0.15
   - Elastic-net mixing parameter. Must be between 0 and 1 (inclusive). l1_ratio=0 corresponds to L2-penalty and l1_ratio=1 corresponds to L1-penalty.
 - solver: string, default='bfgs'
   - Solver method used by `scipy.optimize.minimze` to minimize the loss function. Can be 'bfgs', 'l-bfgs-b', or 'slsqp'.
 - tol: float, default=1e-4
   - Stopping criterion for minimizer. Should be greater than or equal to 0.
 - max_iter: int, default=1000
   - Maximum number of passes over the data during training/minimization. Should be greater than 0.
 - verbose: int, default=0
   - verbosity level. Value other than zero will print convergence messages from minimizer.
 - fit_intercept: bool, default=True
   - Whether an intercept term should be fit in training. It True, a column of ones is concatenated to input data matrix.
 - random_state: int, default=None
   - Seed for randomly initializing coefficients.
 - options: dict, default=None
   - A dictionary of options to pass to solver. 'maxiter' and 'disp' are already included.
### attributes
 - coef_
   - Coefficient fitted to input data features.
 - _estimator_type
   - Type of estimator the instance is ('classifier' or 'regressor').
 - _multi_output
   - Whether the estimator supports multi-output prediction.
 - _check_params
   - Dictionary of parameters to use for validating input data.
### methods
```
set_estimator_type(etype)
```
 - Set type of estimator the instance should be. Can be either 'classifier' or 'regressor'.
```
set_multi_output(multi)
```
 - Set whether the estimator should support multi-output prediction.
```
set_params(**params)
```
 - Set parameters of the estimator.
```
get_params(deep=True)
```
 - Get parameters of the estimator. If deep=True, returns deep copy of parameter dictionary. Else, returns shallow copy.
```
set_check_params(**check_params)
```
 - Set parameters used for validating input data.
```
get_check_params(deep=True)
```
 - Set parameters used for validating input data. If deep=True, returns deep copy of parameter dictionary. Else, returns shallow copy.
```
get_loss_fn()
```
 - Returns loss function used by the estimator. In training, the loss function is combined with the penalty function and is aggregated by sample weights.
```
get_link_fn()
```
 - Returns link function used by the estimator.
```
get_reg_fn()
```
 - Returns penalty/regularization function used by the estimator. In training, the penalty function is combined with the loss function and is multiplied by the strength parameter alpha.
```
partial_fit(X, y, sample_weight=None, **kwargs)
```
 - Perform one training pass on the data. X should be a NxP array of observation data. y should be a Nx1 array of training targets. sample_weight should be a Nx1 array of weights applied to individual samples/observations.
```
fit(X, y, sample_weights=None)
```
 - Perform training on the data until convergence (or max_iter is reached). X should be a NxP array of observation data. y should be a Nx1 array of training targets. sample_weight should be a Nx1 array of weights applied to individual samples/observations.```
predict(X)
```
 - Use the link function to make predictions on the input data. X should be a NxP array of input/observation data.
