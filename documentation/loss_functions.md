___
# mse_loss
```
minimizers.losses.mse_loss(y, y_hat)
```
### Description
Mean squared error loss for single output regression.
### Parameters
 - y: A Nx1 array of continuous valued targets.
 - y_hat: A Nx1 array of continuous valued predictions.
 ### Returns
 - A Nx1 array of computed losses.

___
# multi_mse_loss
```
minimizers.losses.multi_mse_loss(y, y_hat)
```
### Description
Mean squared error loss for multi-output regression.
### Parameters
 - y: A NxK array of continuous valued targets.
 - y_hat: A NxK array of continuous valued predictions.
 ### Returns
 - A Nx1 array of computed losses.

___
# bce_loss
```
minimizers.losses.bce_loss(y, y_hat)
```
### Description
Binary cross entropy loss for binary classification.
### Parameters
 - y: A Nx1 array of binary targets.
 - y_hat: A Nx1 array of probabilisitc predictions.
 ### Returns
 - A Nx1 array of computed losses.

___
# cce_loss
```
minimizers.losses.cce_loss(y, y_hat)
```
### Description
Categorical cross entropy loss for multiple classification.
### Parameters
 - y: A NxK array of categorical targets.
 - y_hat: A NxK array of probabilisitc predictions.
 ### Returns
 - A Nx1 array of computed losses.

___
# hinge_loss
```
minimizers.losses.hinge_loss(y, y_hat)
```
### Description
Hinge loss for binary classification.
### Parameters
 - y: A Nx1 array of binary targets.
 - y_hat: A Nx1 array of margin predictions.
 ### Returns
 - A Nx1 array of computed losses.

___
# mae_loss
```
minimizers.losses.mae_loss(y, y_hat)
```
### Description
Mean absolute error loss for binary classification.
### Parameters
 - y: A Nx1 array of binary targets.
 - y_hat: A Nx1 array of probabilistic predictions.
 ### Returns
 - A Nx1 array of computed losses.

___
# cmae_loss
```
minimizers.losses.cmae_loss(y, y_hat)
```
### Description
Categorical mean absolute error loss for multiple classification.
### Parameters
 - y: A NxK array of categorical targets.
 - y_hat: A NxK array of probabilistic targets.
 ### Returns
 - A Nx1 array of computed losses.

