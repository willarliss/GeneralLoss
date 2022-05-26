___
# zero_penalty
```
minimizers.penalties.zero_penalty(b)
```
### Description
No penalty. Defined for API consistency.
### Parameters
 - b: A (1,P) or (P,) or (K,P) array of coefficients.
 ### Returns
 - The value 0.

___
# l1_penalty
```
minimizers.penalties.l1_penalty(b)
```
### Description
L1-norm penalty.
### Parameters
 - b: A (1,P) or (P,) or (K,P) array of coefficients.
 ### Returns
 - A single float of the computed penalty.

___
# l2_penalty
```
minimizers.penalties.zero_penalty(b)
```
### Description
L2-norm penalty.
### Parameters
 - b: A (1,P) or (P,) or (K,P) array of coefficients.
 ### Returns
 - A single float of the computed penalty.

___
# elasticnet
```
minimizers.penalties.elasticnet(b, gamma=0.5)
```
### Description
Elastic-net penalty.
### Parameters
 - b: A (1,P) or (P,) or (K,P) array of coefficients.
 - gamma: Elastic-net mixing parameter. Must be between 0 and 1 (inclusive). l1_ratio=0 corresponds to L2-penalty and l1_ratio=1 corresponds to L1-penalty.
 ### Returns
 - A single float of the computed penalty.
