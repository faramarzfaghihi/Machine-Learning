# -*- coding: utf-8 -*-
"""
Linear Regression method

@author: faramarz.faghihi
"""
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
x = np.arange(1,10)
y= np.array([28, 25, 26, 31, 32, 29, 30, 35, 36])

plt.scatter(x,y)
plt.show()

x = x.reshape(-1,1)
y = y.reshape(-1,1)
reg = LinearRegression()
reg.fit(x,y)

yhat = reg.predict(x)

fig = plt.figure() 
plt.scatter(x,y)
plt.plot(x,yhat)
plt.show()

from sklearn.datasets import load_boston
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['Price'] = boston.target
boston_df

x = boston.data
y = boston.target

x_train, x_test ,y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
fig = plt.figure()
plt.scatter(y_test, y_pred)
plt.plot()
plt.xlabel('prices')
plt.ylabel('predicted prices')
plt.show()

import sklearn.metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

new_x = boston.data[:,[1,2]]
new_y = boston.target
new_x_train, new_x_test ,new_y_train, new_y_test = train_test_split(new_x, new_y, test_size = 0.3, random_state=42)
new_reg = LinearRegression()
new_reg.fit(new_x_train, new_y_train)
new_y_predict = new_reg.predict(new_x_test)
new_mse = metrics.mean_squared_error(new_y_test, new_y_predict)
new_mse

from sklearn.model_selection import cross_val_score
reg = LinearRegression()
first_cv_scores = cross_val_score(reg, x, y, cv=5)
second_cv_scores = cross_val_score(reg, x, y, cv=10)
print('mean in first_cv_scores is {0:.2f} and in second_cv_scores is {1:.2f}'.format(np
.mean(first_cv_scores), np.mean(second_cv_scores)) )

from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x, y)
lasso_coef = lasso.coef_
print(lasso_coef)
fig = plt.figure()
plt.plot(range(13), lasso_coef)
plt.xticks(range(13), boston.feature_names)
plt.ylabel('Coefficents')
plt.show()



from sklearn.linear_model import Ridge
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(x_train, y_train)
ridge_pred = ridge.predict(x_test)

from sklearn import datasets
bcd = datasets.load_breast_cancer()
x = bcd.data
y = bcd.target

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)
y_prediction = knn.predict(x_test)

print(confusion_matrix(y_test, y_prediction, [0, 1]))
print(classification_report(y_test, y_prediction))

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train, y_train)
y_pred = log.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.preprocessing import normalize
cm = normalize(cm,norm='l1',axis=1)
cm_df = pd.DataFrame(cm, columns=bcd.target_names, index=bcd.target_names)
print(cm_df)

from sklearn.metrics import roc_curve
y_pred_prob = log.predict_proba(x_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
fig = plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_prob)





