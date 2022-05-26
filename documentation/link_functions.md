___
# linear_link
```
minimizers.links.linear_link(X, b)
```
### Description
Linear combination of input matrix and coefficient vector/matrix: <br>
$ \hat{y}_i = X_i b^T $
### Parameters
 - X: A (N,P) array of input data.
 - b: A (1,P) or (P,) or (K,P) array of coefficients.
### Returns
 - A (N,) or (N,K) array of continuous valued predictions.

___
# sigmoid_link
```
minimizers.links.linear_link(X, b)
```
### Description
Sigmoid function applied to linear combination of input matrix and coefficient vector: <br>
$ \hat{p}_i = \frac{1}{1+e^{-X_i b^T}} $
### Parameters
 - X: A (N,P) array of input data.
 - b: A (1,P) or (P,) array of coefficients.
### Returns
 - A (N,) array of probabilisitc predictions.

___
# softmax_link
```
minimizers.links.linear_link(X, B)
```
### Description
Softmax function applied to linear combination of input matrix and coefficient matrix: <br>
$ \hat{P}_i = \frac{e^{X_i B}}{\sum{e^{X_i b^T}}} $
### Parameters
 - X: A (N,P) array of input data.
 - b: A (K,P) array of coefficients.
### Returns
 - A (N,K) array of probabilisitc predictions.
