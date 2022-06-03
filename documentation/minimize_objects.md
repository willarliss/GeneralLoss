___
# `GeneralLossMinimizer`
```
minimizers.minimize.GeneralLossMinimizer(loss_fn=None, link_fn=None, penalty='none', alpha=0.1, l1_ratio=0.15, solver='bfgs', tol=1e-4, max_iter=1000, verbose=0, fit_intercept=True, random_state=None, options=None)
```
### Description
This class is a general minimizer for custom-defined loss functions. Users can specify a custom loss function, custom link function, and custom penalty function. Minimization is performed with Scipy and the API is compatible with Scikit-Learn. The class is designed to support both classification and regression.
### Parameters
 - loss_fn: Callable, default=None
   - Loss function to optimize during training. Should accept an array of targets and an array of predictions. Should return an array of losses. Defaults to binary cross-entropy for _estimator_type='classifier' and _multi_output=False. Defaults to categorical cross-entropy for _estimator_type='classifier' and _multi_output=True. Defaults to mean squared error for _estimator_type='regressor' and _multi_output=False. Defaults to multiple mean squared error for _estimator_type='regressor' and _multi_output=True.
 - link_fn: Callable, default=None
   - Link function to make predictions. Should accept an array of observation data and an array of parameters. Should return an array of predictions. Defaults to the sigmoid function for _estimator_type='classifier' and _multi_output=False. Defaults to softmax function for _estimator_type='classifier' and _multi_output=True. Defaults to linear/identity link function for _estimator_type='regressor'.
 - penalty: Callable or str, default='none'
   - Penalty function to regularize parameters during training. Can be 'none' for no penalty, 'l1' for a L1-penalty, 'l2' for a L2-penalty, 'elasticnet' for an elastic-net penalty, or a callable function. If a function, it should accept an array of coefficients and return a single float. Defaults to no/zero penalty.
 - alpha: float, default=0.1
   - Penalty strength parameter. Constant by which to multiply the penalty function. Should be greater than 0.
 - l1_ratio: float, default=0.15
   - Elastic-net mixing parameter. Must be between 0 and 1 (inclusive). l1_ratio=0 corresponds to L2-penalty and l1_ratio=1 corresponds to L1-penalty.
 - solver: str, default='bfgs'
   - Solver method used by `scipy.optimize.minimze` to minimize the loss function. Can be 'bfgs', 'l-bfgs-b', or 'slsqp'.
 - tol: float, default=1e-4
   - Stopping criterion for minimizer. Should be greater than or equal to 0.
 - max_iter: int, default=1000
   - Maximum number of passes over the data during training/minimization. Should be greater than 0.
 - verbose: int, default=0
   - Verbosity level. Value other than zero will print convergence messages from minimizer.
 - fit_intercept: bool, default=True
   - Whether an intercept term should be fit in training. It True, a column of ones is concatenated to input data matrix.
 - warm_start: bool, default=False
   - When True, use previous solution as initial guess or use manually defined initial guess (initialize_coef). Otherwise, erase the previous solution.
 - random_state: int, default=None
   - Seed for randomly initializing coefficients.
 - options: dict, default=None
   - A dictionary of options to pass to solver. 'maxiter' and 'disp' are already included.
### Attributes
 - coef_
   - Coefficient vector fitted to input data features.
 - n_features_in_
   - Number of features/columns in the input data.
 - n_inputs_
   - Number of input columns (one more than n_features_in_ if fit_intercept=True)
 - n_outputs_
   - Number of targets. In classification, number of classes. In regression, number of output variables.
 - _estimator_type
   - Type of estimator the instance is ('classifier' or 'regressor').
 - _multi_output
   - Whether the estimator supports multi-output prediction.
 - _check_params
   - Dictionary of parameters to use for validating input data.
### Methods
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
 - Get parameters used for validating input data. If deep=True, returns deep copy of parameter dictionary. Else, returns shallow copy.
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
initialize_coef(coef)
```
 - Initialize coefficient array. If nothing is passed, coefficients are initialized normally according to number of inputs and number of outputs.
```
partial_fit(X, y, sample_weight=None, **kwargs)
```
 - Perform one training pass on the data. X should be a (N,P) array of observation data. y should be a (N,1) or (N,) array of training targets. sample_weight should be a (N,1) or (N,) array of weights to be applied to individual samples/observations.
```
fit(X, y, sample_weights=None)
```
 - Perform training on the data until convergence (or max_iter is reached). X should be a (N,P) array of observation data. y should be a (N,1) or (N,) array of training targets. sample_weight should be a (N,1) or (N,) array of weights to be applied to individual samples/observations.
```
decision_function(X)
```
 - Use the link function to make predictions on the input data. X should be a (N,P) array of input/observation data. Returns a (N,) array if _multi_output=False. Returns a (N,K) array if _multi_output=True.
 ```
 predict(X)
 ```
  - Wrapper around decision function.

___
# `CustomLossRegressor`
```
minimizers.minimize.CustomLossRegressor(loss_fn=None, link_fn=None, penalty='none', alpha=0.1, l1_ratio=0.15, solver='bfgs', tol=1e-4, max_iter=1000, verbose=0, fit_intercept=True, random_state=None, options=None)
```
### Description
This class is a minimizer for custom-defined regression loss functions. Users can specify a custom loss function, custom link function, and custom penalty function. Minimization is performed with Scipy and the API is compatible with Scikit-Learn. Outputs/targets will always be assumed to be multi-dimmensional (i.e. a two axis NumPy array).
### Parameters
 - loss_fn: Callable, default=None
   - Loss function to optimize during training. Should accept a (N,K) array of targets and a (N,K) array of predictions. Should return a (N,) array of losses. Defaults to (multiple) mean squared error.
 - link_fn: Callable, default=None
   - Link function to make predictions. Should accept a (N,P) array of observation data and a (K,P) array of parameters. Should return a (N,K) array of predictions. Defaults to linear/identity link function.
 - penalty: Callable or str, default='none'
   - Penalty function to regularize parameters during training. Can be 'none' for no penalty, 'l1' for a L1-penalty, 'l2' for a L2-penalty, 'elasticnet' for an elastic-net penalty, or a callable function. If a function, it should accept a (K,P) array of coefficients and return a single float. Defaults to no/zero penalty.
 - alpha: float, default=0.1
   - Penalty strength parameter. Constant by which to multiply the penalty function. Should be greater than 0.
 - l1_ratio: float, default=0.15
   - Elastic-net mixing parameter. Must be between 0 and 1 (inclusive). l1_ratio=0 corresponds to L2-penalty and l1_ratio=1 corresponds to L1-penalty.
 - solver: str, default='bfgs'
   - Solver method used by `scipy.optimize.minimze` to minimize the loss function. Can be 'bfgs', 'l-bfgs-b', or 'slsqp'.
 - tol: float, default=1e-4
   - Stopping criterion for minimizer. Should be greater than or equal to 0.
 - max_iter: int, default=1000
   - Maximum number of passes over the data during training/minimization. Should be greater than 0.
 - verbose: int, default=0
   - Verbosity level. Value other than zero will print convergence messages from minimizer.
 - fit_intercept: bool, default=True
   - Whether an intercept term should be fit in training. It True, a column of ones is concatenated to input data matrix.
 - random_state: int, default=None
   - Seed for randomly initializing coefficients.
 - warm_start: bool, default=False
   - When True, use previous solution as initial guess or use manually defined initial guess (initialize_coef). Otherwise, erase the previous solution.
 - options: dict, default=None
   - A dictionary of options to pass to solver. 'maxiter' and 'disp' are already included.
### Attributes
 - coef_
   - Coefficient vector fitted to input data features.
 - n_features_in_
   - Number of features/columns in the input data.
 - n_inputs_
   - Number of input columns (one more than n_features_in_ if fit_intercept=True)
 - n_outputs_
   - Number of output variables/targets.
 - _estimator_type
   - Type of estimator the instance is. Automatically set to `regressor'.
 - _multi_output
   - Whether the estimator supports multi-output prediction. Automatically set to True.
 - _check_params
   - Dictionary of parameters to use for validating input data.
### Methods
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
 - Get parameters used for validating input data. If deep=True, returns deep copy of parameter dictionary. Else, returns shallow copy.
```
get_loss_fn()
```
 - Returns loss function used by the estimator. In training, the loss function is combined with the penalty function and is aggregated by sample weights.
```
get_link_fn(wrap=True)
```
 - Returns link function used by the estimator. If wrap is True, returns link function wrapped in multioutput coefficient reshape decorator.
```
get_reg_fn()
```
 - Returns penalty/regularization function used by the estimator. In training, the penalty function is combined with the loss function and is multiplied by the strength parameter alpha.
```
initialize_coef(coef)
```
 - Initialize coefficient array. If nothing is passed, coefficients are initialized normally according to number of inputs and number of outputs.
```
partial_fit(X, y, sample_weight=None, **kwargs)
```
 - Perform one training pass on the data. X should be a (N,P) array of observation data. y should be a (N,1) array of training targets. sample_weight should be a (N,1) or (N,) array of weights to be applied to individual samples/observations.
```
fit(X, y, sample_weights=None)
```
 - Perform training on the data until convergence (or max_iter is reached). X should be a (N,P) array of observation data. y should be a (N,1) array of training targets. sample_weight should be a (N,1) or (N,) array of weights to be applied to individual samples/observations.
```
decision_function(X)
```
 - Use the link function to make predictions on the input data. X should be a (N,P) array of input/observation data. Returns a (N,K) array.
 ```
 predict(X)
 ```
  - Wrapper around decision function. Flattens array to (N,) if single output is (N,1).

___
# `CustomLossClassifier`
```
minimizers.minimize.CustomLossClassifier(loss_fn=None, link_fn=None, penalty='none', alpha=0.1, l1_ratio=0.15, solver='bfgs', tol=1e-4, max_iter=1000, verbose=0, fit_intercept=True, random_state=None, options=None)
```
### Description
This class is a minimizer for custom-defined classification loss functions. Users can specify a custom loss function, custom link function, and custom penalty function. Minimization is performed with Scipy and the API is compatible with Scikit-Learn. Outputs/targets are assumed to always be multi-dimmensional. For binary tasks, the outputs will be of shape Nx2. For K-class tasks, the outputs will be of shape NxK.
### Parameters
 - loss_fn: Callable, default=None
   - Loss function to optimize during training. Should accept a (N,K) array of targets and a (N,K) array of predictions. Should return a (N,) array of losses. Defaults to categorical cross-entropy.
 - link_fn: Callable, default=None
   - Link function to make predictions. Should accept a (N,P) array of observation data and a (K,P) array of parameters. Should return a (N,K) array of predictions. Defaults to softmax function.
 - penalty: Callable or str, default='none'
   - Penalty function to regularize parameters during training. Can be 'none' for no penalty, 'l1' for a L1-penalty, 'l2' for a L2-penalty, 'elasticnet' for an elastic-net penalty, or a callable function. If a function, it should accept a (K,P) array of coefficients and return a single float. Defaults to no/zero penalty.
 - alpha: float, default=0.1
   - Penalty strength parameter. Constant by which to multiply the penalty function. Should be greater than 0.
 - l1_ratio: float, default=0.15
   - Elastic-net mixing parameter. Must be between 0 and 1 (inclusive). l1_ratio=0 corresponds to L2-penalty and l1_ratio=1 corresponds to L1-penalty.
 - solver: str, default='bfgs'
   - Solver method used by `scipy.optimize.minimze` to minimize the loss function. Can be 'bfgs', 'l-bfgs-b', or 'slsqp'.
 - tol: float, default=1e-4
   - Stopping criterion for minimizer. Should be greater than or equal to 0.
 - max_iter: int, default=1000
   - Maximum number of passes over the data during training/minimization. Should be greater than 0.
 - verbose: int, default=0
   - Verbosity level. Value other than zero will print convergence messages from minimizer.
 - fit_intercept: bool, default=True
   - Whether an intercept term should be fit in training. It True, a column of ones is concatenated to input data matrix.
 - random_state: int, default=None
   - Seed for randomly initializing coefficients.
 - warm_start: bool, default=False
   - When True, use previous solution as initial guess or use manually defined initial guess (initialize_coef). Otherwise, erase the previous solution.
 - options: dict, default=None
   - A dictionary of options to pass to solver. 'maxiter' and 'disp' are already included.
### Attributes
 - coef_
   - Coefficient vector fitted to input data features.
 - n_features_in_
   - Number of features/columns in the input data.
 - n_inputs_
   - Number of input columns (one more than n_features_in_ if fit_intercept=True)
 - n_outputs_
   - Number of classes/targets.
 - le_
  - One-hot label encoder transforming 1d array of classes into a 2d array of one-hot encodings.
 - _estimator_type
   - Type of estimator the instance is. Automatically set to `classifier'.
 - _multi_output
   - Whether the estimator supports multi-output prediction. Automatically set to True.
 - _check_params
   - Dictionary of parameters to use for validating input data.
### Methods
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
 - Get parameters used for validating input data. If deep=True, returns deep copy of parameter dictionary. Else, returns shallow copy.
```
get_loss_fn()
```
 - Returns loss function used by the estimator. In training, the loss function is combined with the penalty function and is aggregated by sample weights.
```
get_link_fn(wrap=True)
```
 - Returns link function used by the estimator. If wrap is True, returns link function wrapped in multioutput coefficient reshape decorator.
```
get_reg_fn()
```
 - Returns penalty/regularization function used by the estimator. In training, the penalty function is combined with the loss function and is multiplied by the strength parameter alpha.
```
initialize_coef(coef)
```
 - Initialize coefficient array. If nothing is passed, coefficients are initialized normally according to number of inputs and number of outputs.
```
partial_fit(X, y, sample_weight=None, **kwargs)
```
 - Perform one training pass on the data. X should be a (N,P) array of observation data. y should be a (N,1) or (N,) array of training labels. sample_weight should be a (N,1) or (N,) array of weights to be applied to individual samples/observations.
```
fit(X, y, sample_weights=None)
```
 - Perform training on the data until convergence (or max_iter is reached). X should be a (N,P) array of observation data. y should be a (N,1) or (N,) array of training labels. sample_weight should be a (N,1) or (N,) array of weights to be applied to individual samples/observations.
```
decision_function(X)
```
 - Use the link function to make predictions on the input data. X should be a (N,P) array of input/observation data. Returns a (N,K) array.
 ```
 predict(X)
 ```
  - Wrapper around decision function. Recodes 2d vector into labels and returns a (N,) array.
