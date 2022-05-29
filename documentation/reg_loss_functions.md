___
# mse_loss
```
minimizers.losses.mse_loss(y, y_hat)
```
### Description
Mean squared error loss for single output regression.
### Parameters
 - y: A (N,1) or (N,) array of continuous valued targets.
 - y_hat: A (N,1) or (N,) array of continuous valued predictions.
### Returns
 - A (N,) array of computed losses.

___
# multi_mse_loss
```
minimizers.losses.multi_mse_loss(Y, Y_hat)
```
### Description
Mean squared error loss for multi-output regression.
### Parameters
 - Y: A (N,K) array of continuous valued targets.
 - Y_hat: A (N,K) array of continuous valued predictions.
### Returns
 - A (N,) array of computed losses.

___
# pseudo_huber_loss
```
minimizers.losses.pseudo_huber_loss(y, y_hat, delta=1)
```
### Description
Pseudo-Huber loss for single output regression.
### Parameters
 - y: A (N,1) or (N,) array of continuous valued targets.
 - y_hat: A (N,1) or (N,) array of continuous valued predictions.
 - delta: Smoothing pararmeter.
### Returns
 - A (N,) array of computed losses.

___
# multi_pseudo_huber_loss
```
minimizers.losses.multi_pseudo_huber_loss(Y, Y_hat, delta=1)
```
### Description
Pseudo-Huber loss for multi-output regression.
### Parameters
 - Y: A (N,K) array of continuous valued targets.
 - Y_hat: A (N,K) array of continuous valued predictions.
 - delta: Smoothing pararmeter.
### Returns
 - A (N,) array of computed losses.

___
# gaussian_mle
```
minimizers.losses.gaussian_mle(y, y_hat, scale=1)
```
### Description
Maximum Likelihood Estimation with the Gaussian distribution.
### Parameters
 - y: A (N,1) or (N,) array of continuous valued targets.
 - y_hat: A (N,1) or (N,) array of continuous valued predictions.
 - sigma: Scaling parameter.
### Returns
 - A (N,) array of computed losses.

___
# multivariate_gaussian_mle
```
minimizers.losses.multivariate_gaussian_mle(Y, Y_hat, scale=1)
```
### Description
Maximum Likelihood Estimation with the Multivariate-Gaussian distribution.
### Parameters
 - Y: A (N,K) array of continuous valued targets.
 - Y_hat: A (N,K) array of continuous valued predictions.
 - sigma: Scaling parameter to multiply the covariance matrix by.
### Returns
 - A (N,) array of computed losses.

___
# poisson_mle
```
minimizers.losses.poisson_mle(y, y_hat)
```
### Description
Maximum Likelihood Estimation with the Poisson distribution.
### Parameters
 - y: A (N,1) or (N,) array of discrete valued targets.
 - y_hat: [ndarray] A (N,1) or (N,) array of continuous valued predictions.
### Returns
 - A (N,) array of computed losses.
