___
# linear_link
```
minimizers.links.linear_link(X, b)
```
### Description
Linear combination of input matrix and coefficient vector: $\hat{y}_i = X_i b^T$.
### Parameters
 - X: pass
 - b: pass
 ### Returns
 - pass

___
# multi_linear_link
```
minimizers.links.multi_linear_link(X, B)
```
### Description
Linear combination of input matrix and coefficient matrix: $\hat{Y}_i = X_i B$.
### Parameters
 - X: pass
 - B: pass
 ### Returns
 - pass

___
# sigmoid_link
```
minimizers.links.linear_link(X, b)
```
### Description
Sigmoid function applied to linear combination of input matrix and coefficient vector: $\hat(p) = \frac{1}{1+e^{-X_i b^T}}$.
### Parameters
 - X: pass
 - b: pass
 ### Returns
 - pass

___
# softmax_link
```
minimizers.links.linear_link(X, B)
```
### Description
Softmax function applied to linear combination of input matrix and coefficient matrix: $\hat{P}_i = \frac{e^{X_i B}}{\sum{e^{X_i B}}} $
### Parameters
 - X: pass
 - B: pass
 ### Returns
 - pass
