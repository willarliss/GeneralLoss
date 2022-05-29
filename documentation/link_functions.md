___
# linear_link
```
minimizers.links.linear_link(X, b)
```
### Description
Linear combination of input matrix and coefficient vector/matrix.
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
Sigmoid function applied to linear combination of input matrix and coefficient vector.
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
Softmax function applied to linear combination of input matrix and coefficient matrix.
### Parameters
 - X: A (N,P) array of input data.
 - b: A (K,P) array of coefficients.
### Returns
 - A (N,K) array of probabilisitc predictions.

___
# log_link
```
minimizers.links.log_link(X, B)
```
### Description
Log-link function applied to linear combination of input matrix and coefficient vector/matrix.
### Parameters
 - X: A (N,P) array of input data.
 - b: A (1,P) or (P,) or (K,P) array of coefficients.
### Returns
 - A (N,) or (N,K) array of continuous valued predictions.

___
# inverse_link
```
minimizers.links.inverse_link(X, B)
```
### Description
Inverse-link function applied to linear combination of input matrix and coefficient vector/matrix.
### Parameters
 - X: A (N,P) array of input data.
 - b: A (1,P) or (P,) or (K,P) array of coefficients.
### Returns
 - A (N,) or (N,K) array of continuous valued predictions.
