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
 - _estimator_type
 - _multi_output
 - _check_params
### methods
'''
set_estimator_type(etype)
'''
 - pass
```
get_params
```
 - pass
```
set_params
```
 - pass
```
set_check_params
```
 - pass
```
get_check_params
```
 - pass
```
get_loss_fn
```
 - pass
```
get_link_fn
```
 - pass
```
get_reg_fn
```
 - pass
```
partial_fit
```
 - pass
```
fit
```
 - pass
```
predict
```
 - pass
