# -*- coding: utf-8 -*-
"""
@author: ALA
Script for principal component analysis
"""

%load_ext autoreload
%autoreload 2

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# Import chempy and numpy packages
import chempy as cp
import numpy as np


# Let's import some data
# First, import spectral data
X = cp.read2div('./data_set/X1.CSV')
# Then, import Y values
Y = cp.read2div('./data_set/Y1.CSV')

# Cut X/Y in two parts
X1 = cp.selectrow(X, list(np.arange(0,200)))
X2 = cp.selectrow(X, list(np.arange(200,X.d.shape[0])))

Y1 = cp.selectrow(Y, list(np.arange(0,200)))
Y2 = cp.selectrow(Y, list(np.arange(200,X.d.shape[0])))

Y1s = cp.selectcol(Y, [19])
Y2s = cp.selectcol(Y2, [19])



# Perform PLS with the single Y case
pls_single, pred = cp.pls(X,Y1s,11)


cp.curve(cp.transpose(Y1s))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(Y1s.d)
plt.show()

# Look at the Beta for Lv = 5
cp.curve(pls_single.beta, row=[1])
# Look at PLS Components
cp.curve(pls_single.W, row=[7])
# Look at PLS Components
cp.map2(pls_single.T, 0,1)
# Look at training performance (R2)
cp.curve(pred.perf, row=[1],legend_label=pred.perf.v[0])

# Look at vip
cp.curve(cp.transpose(pls_single.vip), row=[0])

# Apply PLS on test set
pred_test = pls_single.predict(X2, Y2s)

# Look at training performance (RMSE)
cp.curve(pred_test.perf, row=[1],legend_label=pred_test.perf.v[0])

##### COMMENTS
## Legend should work here
## Should be able to draw map from two different div

# Look at PLS T scores
pls_multi, pred = cp.pls(X1,Y1,10)



# Look at the Beta for Lv = 5 for the 20th variable
cp.curve(pls_multi.beta[19], row=4)

# Look at the VIP for Lv = 5 for the 20th variable
cp.curve(cp.transpose(pls_multi.vip[0]))



# Perform Ridge Regression
Tvec = [0.001, 0.1, 1, 10, 100]
ridge_multi, pred = cp.ridge_regression(X1,Y1,Tvec)
ridge_single, pred = cp.ridge_regression(X1,Y1s,Tvec)
