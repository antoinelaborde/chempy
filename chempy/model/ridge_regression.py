# -*- coding: utf-8 -*-
"""
Created on 19/10/2018

@author: ALA
"""

import numpy as np
import chempy.utils.util as util
import chempy.utils.classes as classes


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
    
    Credits
    -------

    """
    if x_div.d.shape[0] != y_div.d.shape[0]:
        raise ValueError('x_div and y_div must have the same number of rows')
    
    ridge_obj = RidgeRegression()
    ridge_obj.fit(x_div, y_div, T_range, centred)
    pred = ridge_obj.predict(x_div, y_div)

    return ridge_obj, pred

    
class RidgeRegression(classes.Foo):

    def fit(self,x_div, y_div, T_range, centred=True):
        """
        performs ridge regression of x_div on y_div 
        Parameters
        ----------
        x_div: div structure (mandatory)
            div containing data
        y_div: div structure (mandatory)
            div reference values
        T_range: list (mandatory)
            list of float for the Tikhonov matrix identity element
        """
        self.x_div = util.copy(x_div)
        self.y_div = util.copy(y_div)
        self.centred = centred
        self.T = T_range

        X = self.x_div.d
        Y = self.y_div.d.astype(float)

        Beta, mean_y, mean_x = ridge(X, Y, T_range, centred=centred)

        beta_list = []
        for i in range(Beta.shape[1]):
            beta = util.Div(d=Beta[:,i,:].T, v=x_div.v, id='beta ' + self.y_div.v[i], i=['T=' + str(T) for T in T_range])
            beta_list.append(beta)
        if len(beta_list) == 1:
            self.beta = beta_list[0]
        else:
            self.beta = beta_list

        self.mean_x = util.Div(d=np.expand_dims(mean_x, axis=1).T, v=self.x_div.v,id='Mean X')
        self.mean_y = util.Div(d=mean_y, id='Mean Y')

    def predict(self, x_div, ytrue_div):
        """
        apply partial least square regression on x_div
        Parameters
        ----------
        x_div: div structure (mandatory)
            div containing data
        ytrue_div: div structure (mandatory)
            div reference values
        """
        # n: number of individuals
        # p: number of Y values to predict per indiv
        # b: number of latent variables
        n = x_div.d.shape[0]
        if isinstance(self.beta, list):
            p = len(self.beta)
            b = self.beta[0].d.shape[0]
        else:
            p = 1
            b = self.beta.d.shape[0]

        # One Y case
        if p == 1:
            yhat = np.zeros((n, b))
            perf = np.zeros((4, b))
            beta = self.beta.d
            for j in range(b):
                yhat[:,j] = np.dot(x_div.d - self.mean_x.d, beta[j,:]) + self.mean_y.d
                calc_perf = util.quantif_perf(ytrue_div.d, np.expand_dims(yhat[:,j],axis=1), nb_variables=j)
                perf[:,j] = np.array(list(calc_perf.values()))
   
            yh = util.Div(d=yhat, i=x_div.i,id='Yhat ' + self.y_div.v[0], v=['T=' + str(T) for T in self.T])

            perf = util.Div(d=perf, i=list(calc_perf.keys()),id='Performance Ridge ' + self.y_div.v[0], v=['T=' + str(T) for T in self.T])

        # Multi Y case
        else:
            yhat_list = []
            perf_list = []
            for i in range(p):
                yhat = np.zeros((n, b))
                perf = np.zeros((4, b))
                beta = self.beta[i].d
                for j in range(b):
                    yhat[:,j] = np.dot(x_div.d - self.mean_x.d,beta[j,:]) + self.mean_y.d[j]

                    calc_perf = util.quantif_perf(ytrue_div.d[:,i], yhat[:,j], nb_variables=j)
                    perf[:,j] = np.array(list(calc_perf.values()))

                yh = util.Div(d=yhat, i=x_div.i,id='Yhat ' + self.y_div.v[i], v=['T=' + str(T) for T in self.T])

                perf = util.Div(d=perf, i=list(calc_perf.keys()),id='Performance Ridge ' + self.y_div.v[i], v=['T=' + str(T) for T in self.T])

                yhat_list.append(yh)
                perf_list.append(perf)


        pred_obj = util.Foo(predy = yh, perf = perf)
        return pred_obj



def ridge(X, Y, T, centred=True):
    """
    ridge algorithm 
    Parameters
    ----------
    X: numpy array (mandatory)
        predictors
    Y: numpy array (mandatory)
        target values
    T: list of float (mandatory)
        Tikhonov matrix identity element

    Returns
    -------

    """
    n, p = X.shape
    _, q = Y.shape


    if centred:
        mean_x = np.mean(X, axis=0)
        mean_y = np.mean(Y, axis=0)   
    else:
        mean_x = np.zeros((p,1))
        mean_y = np.zeros((q,1))
    X = X - mean_x
    Y = Y - mean_y


    # Compute matrix product out of the loop for speed
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, Y)

    # Initialize matrices
    Beta = np.zeros((p, q, len(T)))

    # Evaluate ridge regression Beta coeff for all values of Tikhonov regularization value (t)
    for i, t in enumerate(T):
        beta = np.dot(np.linalg.inv(XtX + t * np.identity(p)), XtY)
        Beta[:,:,i] = beta

    return Beta, mean_y, mean_x