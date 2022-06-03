import os
from functools import partial

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification, make_regression

from minimizers import links
from minimizers import losses
from minimizers.minimize import GeneralLossMinimizer, CustomLossClassifier, CustomLossRegressor

seed = 0

### Binary classification

X, y = make_classification(
    n_classes=2,
    n_samples=1000,
    n_features=10,
    flip_y=0.,
    n_clusters_per_class=1,
    random_state=seed,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
del X, y

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.bce_loss,
        link_fn=links.sigmoid_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(False)

model.fit(X_train, y_train)
y_hat = (model.predict(X_train)>0.5).astype(int).flatten()
y_hat = (model.predict(X_test)>0.5).astype(int).flatten()

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossClassifier(
        loss_fn=losses.cce_loss,
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.mae_loss,
        link_fn=links.sigmoid_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(False)

model.fit(X_train, y_train)
y_hat = (model.predict(X_train)>0.5).astype(int).flatten()
y_hat = (model.predict(X_test)>0.5).astype(int).flatten()

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossClassifier(
        loss_fn=losses.cmae_loss,
        link_fn=links.softmax_link,
        random_state=seed,
        tol=1e-3,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

lambda_ = 0.5

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=partial(losses.neg_box_cox_loss, lam=lambda_),
        link_fn=links.sigmoid_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(False)

model.fit(X_train, y_train)
y_hat = (model.predict(X_train)>0.5).astype(int).flatten()
y_hat = (model.predict(X_test)>0.5).astype(int).flatten()

lambda_ = 0.5

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossClassifier(
        loss_fn=partial(losses.multi_neg_box_cox_loss, lam=lambda_),
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.binomial_mle,
        link_fn=links.sigmoid_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(False)

model.fit(X_train, y_train)
y_hat = (model.predict(X_train)>0.5).astype(int).flatten()
y_hat = (model.predict(X_test)>0.5).astype(int).flatten()

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossClassifier(
        loss_fn=losses.multinomial_mle,
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

power = 1.

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=partial(losses.hinge_loss, power=power),
        link_fn=links.linear_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(False)

yp_train = np.where(y_train==0, -1, 1)
yp_test = np.where(y_test==0, -1, 1)
model.fit(X_train, yp_train)
y_hat = np.where(model.predict(X_train)>0., 1, -1).flatten()
y_hat = np.where(model.predict(X_test)>0., 1, -1).flatten()

power = 1.

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.perceptron_loss,
        link_fn=links.sigmoid_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(False)

yp_train = np.where(y_train==0, -1, 1)
yp_test = np.where(y_test==0, -1, 1)
model.fit(X_train, yp_train)
y_hat = np.where(model.predict(X_train)>0.5, 1, -1).flatten()
y_hat = np.where(model.predict(X_test)>0.5, 1, -1).flatten()

### Multi-class classification

X, y = make_classification(
    n_classes=3,
    n_samples=1000,
    n_features=10,
    flip_y=0.,
    n_clusters_per_class=1,
    random_state=seed,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
del X, y

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.cce_loss,
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(True)

ohe = OneHotEncoder(sparse=False)
ye_train = ohe.fit_transform(y_train.reshape(-1,1))
ye_test = ohe.transform(y_test.reshape(-1,1))

model.fit(X_train, ye_train)
y_hat = model.predict(X_train).argmax(1)
y_hat = model.predict(X_test).argmax(1)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossClassifier(
        loss_fn=losses.cce_loss,
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.cmae_loss,
        link_fn=links.softmax_link,
        random_state=seed,
        tol=5e-3
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(True)

ohe = OneHotEncoder(sparse=False)
ye_train = ohe.fit_transform(y_train.reshape(-1,1))
ye_test = ohe.transform(y_test.reshape(-1,1))

model.fit(X_train, ye_train)
y_hat = model.predict(X_train).argmax(1)
y_hat = model.predict(X_test).argmax(1)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossClassifier(
        loss_fn=losses.cmae_loss,
        link_fn=links.softmax_link,
        random_state=seed,
        tol=5e-3,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

lambda_ = 0.5

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=partial(losses.multi_neg_box_cox_loss, lam=lambda_),
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(True)

ohe = OneHotEncoder(sparse=False)
ye_train = ohe.fit_transform(y_train.reshape(-1,1))
ye_test = ohe.transform(y_test.reshape(-1,1))

model.fit(X_train, ye_train)
y_hat = model.predict(X_train).argmax(1)
y_hat = model.predict(X_test).argmax(1)

lambda_ = 0.5

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossClassifier(
        loss_fn=partial(losses.multi_neg_box_cox_loss, lam=lambda_),
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.multinomial_mle,
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('classifier')
model[-1].set_multi_output(True)

ohe = OneHotEncoder(sparse=False)
ye_train = ohe.fit_transform(y_train.reshape(-1,1))
ye_test = ohe.transform(y_test.reshape(-1,1))

model.fit(X_train, ye_train)
y_hat = model.predict(X_train).argmax(1)
y_hat = model.predict(X_test).argmax(1)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossClassifier(
        loss_fn=losses.multinomial_mle,
        link_fn=links.softmax_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

### Single output regression

X, y = make_regression(
    n_targets=1,
    n_samples=1000,
    n_features=10,
    random_state=seed,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
del X, y

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.mse_loss,
        link_fn=links.linear_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('regressor')
model[-1].set_multi_output(False)

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossRegressor(
        loss_fn=losses.multi_mse_loss,
        link_fn=links.linear_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

delta = 1.

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=partial(losses.pseudo_huber_loss, delta=delta),
        link_fn=links.linear_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('regressor')
model[-1].set_multi_output(False)

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

delta = 1.

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossRegressor(
        loss_fn=partial(losses.multi_pseudo_huber_loss, delta=delta),
        link_fn=links.linear_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.gaussian_mle,
        link_fn=links.linear_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('regressor')
model[-1].set_multi_output(False)

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossRegressor(
        loss_fn=losses.multivariate_gaussian_mle,
        link_fn=links.linear_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.poisson_mle,
        link_fn=links.log_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('regressor')
model[-1].set_multi_output(False)

yp_train = np.exp(y_train*0.01).round()
yp_test = np.exp(y_test*0.01).round()
model.fit(X_train, yp_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

### Multi-output regression

X, y = make_regression(
    n_targets=2,
    n_samples=1000,
    n_features=10,
    random_state=seed,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
del X, y

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.multi_mse_loss,
        link_fn=links.linear_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('regressor')
model[-1].set_multi_output(True)

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossRegressor(
        loss_fn=losses.multi_mse_loss,
        link_fn=links.linear_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

delta = 1.

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=partial(losses.multi_pseudo_huber_loss, delta=delta),
        link_fn=links.linear_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('regressor')
model[-1].set_multi_output(True)

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

delta = 1.

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossRegressor(
        loss_fn=partial(losses.multi_pseudo_huber_loss, delta=delta),
        link_fn=links.linear_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', GeneralLossMinimizer(
        loss_fn=losses.multivariate_gaussian_mle,
        link_fn=links.linear_link,
        random_state=seed,
    )),
])
model[-1].set_estimator_type('regressor')
model[-1].set_multi_output(True)

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)

model = Pipeline([
    ('sc', StandardScaler()),
    ('est', CustomLossRegressor(
        loss_fn=losses.multivariate_gaussian_mle,
        link_fn=links.linear_link,
        random_state=seed,
    )),
])

model.fit(X_train, y_train)
y_hat = model.predict(X_train)
y_hat = model.predict(X_test)
