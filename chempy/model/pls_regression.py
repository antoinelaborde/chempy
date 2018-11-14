# -*- coding: utf-8 -*-
"""
Created on 19/10/2018

@author: ALA
"""

import numpy as np
import chempy.utils.util
import scipy

"""
def pls(div, normed=False, centred=True):
PLS EN TRAVAUX

"""

def pls_regression(x_div, y_div, component_range):
    """
    performs partial least square regression of data in div with respect to y values
    Parameters
    ----------
    x_div: div structure (mandatory)
        div containing data
    y_div: div structure (mandatory)
        div reference values
    component_range: list (mandatory)
        list of int for the number of components to use

    Returns
    -------
    object with attributes:

    Notes
    -----
    Model of PLS:
    X = TP' + E
    Y = TQ' + F
    where:
        T: scores of X
        U: scores of Y
        W: weights of X
        C: weights of Y
        P: loadings of X
        Q: loadings of Y
        E: error matrix
        F: error matrix

        T[:, k] = Xk W[:, k]
        U[:, k] = Yk C[:, k]
        where Xk and Yk are residual matrices at step k.
    PLS is performed using SIMPLS algorithm (with SVD)
    
    Version Dominique ≠ de celle de sklearn:
    -pas de standardization (besoin de rajouter en option?)
    -pas le meme calcul des scores/loadings etc.


    Credits
    -------

    """
    X = x_div.d
    Y = y_div.d.astype(float)
    n, p = X.shape
    _, q = Y.shape
    n_component = len(component_range)+1
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


    Xk = X
    Yk = Y
    # Version domi
    beta = np.zeros((p, q))
    # Loop on components
    for i,k in enumerate(component_range):
        # Weights estimation with SVD
        Usvd, _, Vsvd = scipy.linalg.svd(np.dot(Xk.T,Yk), full_matrices=False)
        w = Usvd[:, 0]
        # Ensure output of svd to be deterministic
        w = svd_corr(w)

        # X scores computation
        t = np.dot(Xk, w)
        # Compute components by regression
        # c are coefficients such that y = c*t
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

        # Compute beta
        #beta = np.dot(x_loadings, Q.T)
        # Store beta
        #Beta[:,:,k] = beta
        Beta[:,:,i] = beta

    # Compute loadings to have
    # T = X W*  where W* = x_loadings
    # U = Y C*  where C* = y_loadings
    #x_loadings = np.dot(W,scipy.linalg.pinv2(np.dot(P.T, W),check_finite=False))
    #y_loadings = np.dot(C,scipy.linalg.pinv2(np.dot(Q.T, C),check_finite=False))
    Beta = np.reshape(Beta, (Beta.shape[0], Beta.shape[2]))


    return Beta
        


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
    x *= np.sign(max_abs_cols)
    return x