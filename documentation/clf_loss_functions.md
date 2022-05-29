___
# bce_loss
```
minimizers.losses.bce_loss(y, y_hat)
```
### Description
Binary cross entropy loss for binary classification.
### Parameters
 - y: A (N,1) or (N,) array of binary targets.
 - y_hat: A (N,1) or (N,) array of probabilisitc predictions.
### Returns
 - A (N,) array of computed losses.

___
# cce_loss
```
minimizers.losses.cce_loss(Y, Y_hat)
```
### Description
Categorical cross entropy loss for multiple classification.
### Parameters
 - Y: A (N,K) array of categorical targets.
 - Y_hat: A (N,K) array of probabilisitc predictions.
### Returns
 - A (N,) array of computed losses.

___
# mae_loss
```
minimizers.losses.mae_loss(y, y_hat)
```
### Description
Mean absolute error loss for binary classification.
### Parameters
 - y: A (N,1) or (N,) array of binary targets.
 - y_hat: A (N,1) or (N,) array of probabilistic predictions.
### Returns
 - A (N,) array of computed losses.

___
# cmae_loss
```
minimizers.losses.cmae_loss(Y, Y_hat)
```
### Description
Categorical mean absolute error loss for multiple classification.
### Parameters
 - Y: A (N,K) array of categorical targets.
 - Y_hat: A (N,K) array of probabilistic targets.
### Returns
 - A (N,) array of computed losses.

---
# neg_box_cox_loss
```
minimizers.losses.neg_box_cox_loss(y, y_hat, lam=1)
```
### Description
Negative Box-Cox transformation function for binary classification.
### Parameters
 - y: A (N,1) or (N,) array of binary targets.
 - y_hat: A (N,1) or (N,) array of probabilistic predictions.
 - lam: Power term in Box-Cox transform.
### Returns
 - A (N,) array of computed losses.

---
# multi_neg_box_cox_loss
```
minimizers.losses.multi_neg_box_cox_loss(Y, Y_hat, lam=1)
```
### Description
Negative Box-Cox transformation function for categorical classification.
### Parameters
 - Y: A (N,K) array of categorical targets.
 - Y_hat: A (N,K) array of probabilistic predictions.
 - lam: Power term in Box-Cox transform.
### Returns
 - A (N,) array of computed losses.

---
# binomial_mle
```
minimizers.losses.binomial_mle(y, y_hat)
```
### Description
Maximum Likelihood Estimation with the Binomial distribution.
### Parameters
 - y: A (N,1) or (N,) array of binary targets.
 - y_hat: A (N,1) or (N,) array of probabilistic predictions.
### Returns
 - A (N,) array of computed losses.

---
# multinomial_mle
```
minimizers.losses.multinomial_mle(Y, Y_hat)
```
### Description
Maximum Likelihood Estimation with the Multinomial distribution.
### Parameters
 - Y: A (N,K) array of binary targets.
 - Y_hat: A (N,K) array of probabilistic predictions.
### Returns
 - A (N,) array of computed losses.

---
# perceptron_loss
```
minimizers.losses.perceptron_loss(y, y_hat)
```
### Description
Perceptron loss for binary classification.
### Parameters
 - y: A (N,1) or (N,) array of +/- targets.
 - y_hat: A (N,1) or (N,) array of probabilistic predictions.
### Returns
 - A (N,) array of computed losses.

---
# hinge_loss
```
minimizers.losses.hinge_loss(y, y_hat, power=1)
```
### Description
Hinge loss for binary classification. Optional squared or p-degree.
### Parameters
 - y: A (N,1) or (N,) array of +/- targets.
 - y_hat: A (N,1) or (N,) array of margin predictions.
 - power: Power to raise hinge loss by.
### Returns
 - A (N,) array of computed losses.
