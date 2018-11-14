# -*- coding: utf-8 -*-
"""
Created on 19/10/2018

@author: ALA
"""

import numpy as np
import chempy.utils.util as util


"""
def ridge_regression(x_div, y_div, T_range, centred=True):
"""

def ridge_regression(x_div, y_div, T_range, centred=True):
    """
    performs ridge regression of data in div with respect to y values
    Parameters
    ----------
    x_div: div structure (mandatory)
        div containing data
    y_div: div structure (mandatory)
        div reference values
    T_range: list (mandatory)
        list of float for the Tikhonov matrix identity element
    centred: bool (optional, default=True)
        if True centers Y and X

    Returns
    -------
    object with attributes:
        info: structure
        beta: div
        mean_x: div
        mean_y: div
        rmsec: div
        r2c: div
    
    Credits
    -------

    """
    X = x_div.d
    Y = y_div.d.astype(float)
    n, p = X.shape

    if centred:
        mean_x = np.mean(X, axis=0)
        mean_y = np.mean(Y, axis=0)
        X = X - mean_x
        Y = Y - mean_y
    else:
        mean_x = np.zeros((p,1))
        mean_y = 0

    # Compute matrix product out of the loop for speed
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, Y)

    # Initialize matrices
    Beta_coeff = np.zeros((p, len(T_range)))
    Predy = np.zeros((n, len(T_range)))
    Error = np.zeros((n, len(T_range)))
    Rmsec = np.zeros((len(T_range)))
    R2c = np.zeros((len(T_range)))

    # Evaluate ridge regression Beta coeff for all values of Tikhonov regularization value (t)
    for i, t in enumerate(T_range):
        Beta_coeff[:, i] = np.dot(np.linalg.inv(XtX + t * np.identity(p)), XtY).T
        Predy[:, i] = np.dot(X, Beta_coeff[:,i]) + mean_y
        Error[:, i] = Predy[:, i]  - np.ravel(Y)
        Rmsec[i] = np.sqrt(np.mean(np.square(Error[:,i])))
        # Calculation for R2
        y_corr = np.corrcoef(np.ravel(Y), Predy[:,i])
        R2c[i] = np.square(y_corr[0,1])

    axisname = ['T=' + str(t) for t in T_range]

    # Outputs
    beta_div = util.Div(d=Beta_coeff, i=x_div.v, v=axisname)
    xmean_div = util.Div(d=mean_x, i='mean x', v=x_div.v)
    ymean_div = util.Div(d=mean_y, i='mean y', v=y_div.v)
    rmsec_div = util.Div(d=Rmsec, i='RMSEC', v=axisname)
    r2c_div = util.Div(d=R2c, i='R2C', v=axisname)
    predy_div = util.Div(d=Predy, i=x_div.i, v=axisname)

    info_obj = util.Foo(type='ridge_regression', centred=centred)

    ridge_regression_obj = util.Foo(info=info_obj, beta=beta_div, mean_x=xmean_div, mean_y=ymean_div, rmsec=rmsec_div, r2c=r2c_div, predy=predy_div)

    return ridge_regression_obj