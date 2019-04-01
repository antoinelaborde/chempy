# -*- coding: utf-8 -*-
"""
Created on 19/10/2018

@author: ALA
"""

import numpy as np
import chempy.utils.util as util
import chempy.utils.classes as classes

import scipy

"""
def pls():

"""

def pls(x_div, y_div, max_dim):
    """
    performs partial least square regression of x_div on y_div 
    Parameters
    ----------
    x_div: div structure (mandatory)
        div containing data
    y_div: div structure (mandatory)
        div reference values
    max_dim: int (mandatory)
        maximum numer of latent variable to use


    Returns
    -------
    Pls class instance
    
    Credits
    -------
    Dominique Bertrand: dataframe@free.fr
    Antoine Laborde: antoine.laborde@greentropism.com
    """

    if x_div.d.shape[0] != y_div.d.shape[0]:
        raise ValueError('x_div and y_div must have the same number of rows')

    pls_obj = Pls()
    pls_obj.fit(x_div, y_div, max_dim)
    pred = pls_obj.predict(x_div, y_div)


    return pls_obj, pred



class Pls(classes.Foo):

    def fit(self, x_div, y_div, max_dim):
        """
        performs partial least square regression of x_div on y_div 
        Parameters
        ----------
        x_div: div structure (mandatory)
            div containing data
        y_div: div structure (mandatory)
            div reference values
        max_dim: int (mandatory)
            maximum numer of latent variable to use
        """
        self.x_div = util.copy(x_div)
        self.y_div = util.copy(y_div)

        X = self.x_div.d
        Y = self.y_div.d.astype(float)

        Beta, Beta0, T, P, Q, W = pls2(X, Y, max_dim)

        # beta_list contains one div structure for each y value to predict
        beta_list = []
        beta0_list = []
        for i in range(Beta.shape[1]):
            beta = util.Div(d=Beta[:,i,:], v=x_div.v,id='beta ' + self.y_div.v[i], i=[str(i) + ' latent variables' for i in range(1,max_dim+1)])

            beta0 = util.Div(d=np.expand_dims(Beta0[i,:],axis=1), v=['Beta0'],id='beta 0 ' + self.y_div.v[i], i=[str(i) + ' latent variables' for i in range(1,max_dim+1)])

            beta_list.append(beta)
            beta0_list.append(beta0)
        if len(beta_list) == 1:
            self.beta = beta_list[0]
            self.beta0 = beta0_list[0]
        else:
            self.beta = beta_list
            self.beta0 = beta0_list
        

        self.W = util.Div(d=W.T, v=x_div.v,id='W', i=['pls ' + str(i) for i in range(1,max_dim+1)])

        self.T = util.Div(d=T, i=x_div.i,id='T', v=['pls ' + str(i) for i in range(1,max_dim+1)])

        self.Q = util.Div(d=Q,id='Q')

        self.P = util.Div(d=P, i=x_div.v,id='P', v=[str(i) + ' latent variables' for i in range(1,max_dim+1)])

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
                yhat[:,j] = np.dot(x_div.d, beta[j,:]) + self.beta0.d[j]
                calc_perf = util.quantif_perf(ytrue_div.d, np.expand_dims(yhat[:,j],axis=1), nb_variables=j)
                perf[:,j] = np.array(list(calc_perf.values()))
   
            yh = util.Div(d=yhat, i=x_div.i,id='Yhat ' + self.y_div.v[0], v=[str(i) + ' latent variables' for i in range(1,b+1)])

            perf = util.Div(d=perf, i=list(calc_perf.keys()),id='Performance PLS ' + self.y_div.v[0], v=[str(i) + ' latent variables' for i in range(1,b+1)])

        # Multi Y case
        else:
            yhat_list = []
            perf_list = []
            for i in range(p):
                yhat = np.zeros((n, b))
                perf = np.zeros((4, b))
                beta = self.beta[i].d
                for j in range(b):
                    yhat[:,j] = np.dot(x_div.d, beta[j,:]) + self.beta0[i].d[j]

                    calc_perf = util.quantif_perf(ytrue_div.d[:,i], yhat[:,j], nb_variables=j)
                    perf[:,j] = np.array(list(calc_perf.values()))

                yh = util.Div(d=yhat, i=x_div.i,id='Yhat ' + self.y_div.v[i], v=[str(i) + ' latent variables' for i in range(1,b+1)])

                perf = util.Div(d=perf, i=list(calc_perf.keys()),id='Performance PLS ' + self.y_div.v[i], v=[str(i) + ' latent variables' for i in range(1,b+1)])

                yhat_list.append(yh)
                perf_list.append(perf)

            yh = yhat_list
            perf = perf_list
        
        pred_obj = util.Foo(predy = yh, perf = perf)

        return pred_obj


def pls2(X, Y, maxdim):
    """
    pls2 algorithm for PLS on Y vector
    Parameters
    ----------
    X: numpy array (mandatory)
        predictors
    Y: numpy array (mandatory)
        target values
    max_dim: int (mandatory)
        maximum numer of latent variable to use

    Returns
    -------

    """
    n, p = X.shape
    _, q = Y.shape
    n_component = maxdim
    component_range = np.arange(maxdim)
    # Center X and Y
    mean_x = np.mean(X, axis=0)
    mean_y = np.mean(Y, axis=0)

    X -= mean_x
    Y -= mean_y

    # Initialize matrices
    T = np.zeros((n, n_component))
    U = np.zeros((n, n_component))
    W = np.zeros((p, n_component))
    C = np.zeros((q, n_component))
    P = np.zeros((p, n_component))
    Q = np.zeros((q, n_component))
    Beta = np.zeros((p, q, n_component))
    Beta0 = np.zeros((q, n_component))

    Xk = X
    Yk = Y
    beta = np.zeros((p, q))
    # Loop on components
    for i, k in enumerate(component_range):
        # Weights estimation with SVD
        Usvd, _, Vsvd = scipy.linalg.svd(np.dot(Xk.T,Yk), full_matrices=False)
        w = Usvd[:, 0]

        # Ensure output of svd to be deterministic
        w = svd_corr(w)
        # X scores computation
        t = np.dot(Xk, w)
        # Compute components by regression
        # Q are coefficients such that y = q*t
        # p are coefficients such that x = p*t
        q = np.dot(Yk.T, t) / np.dot(t.T, t)
        p = np.dot(Xk.T, t) / np.dot(t.T, t)
        
        t = np.expand_dims(t, axis=1)
        q = np.expand_dims(q, axis=1)
        p = np.expand_dims(p, axis=1)
        w = np.expand_dims(w, axis=1)
        # Deflation
        Xk -= np.dot(t, p.T)
        Yk -= np.dot(t, q.T)

        # Storing
        T[:, k] = t.ravel()
        P[:, k] = p.ravel()
        Q[:, k] = q.ravel()
        W[:, k] = w.ravel()
        # Compute Beta
        beta += np.dot(w, q.T)
        beta0 = mean_y - np.dot(mean_x, beta)
        # Store beta
        Beta[:,:,i] = beta
        Beta0[:,i] = beta0
    return Beta.T, Beta0, T, P, Q, W

def svd_corr(x):
    """
    corrects the sign of svd u and v output
    Parameters
    ----------
    x: ndarray (mandatory)
        x output of svd

    Returns
    -------

    Notes
    -----

    Credits
    -------

    """
    max_abs_cols = np.argmax(np.abs(x), axis=0)
    x *= np.sign(x[max_abs_cols])
    return x