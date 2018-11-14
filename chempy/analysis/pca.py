# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:00:54 2015

@author: DOMI
Updated by ALA 2018/10/10
"""

import numpy as np
import chempy.utils.util


"""
def pca(div, normed=False, centred=True):
"""

def pca(div, normed=False, centred=True):
    """
    perform principal component analysis (PCA) on a div structure
    Parameters
    ----------
    div: div structure (mandatory)
        div containing data
    normed: bool (optional, default=False)
        norm data before PCA
    centred: bool (optional, default=True)
        center data before PCA

    Returns
    -------
    object with attributes:
        info: object
        score: div
        supscore: div
        eigenvec: div
        eigenval: div
        mean: div
        std: div
        varscore: div
    
    Credits
    -------
    Dominique Bertrand: dataframe@free.fr
    """

    X = div.d
    n, p = X.shape

    if centred:
        mean_x = np.mean(X, axis=0)
        X = X - mean_x
        mean_div = util.Div(d=mean_x, i='mean', v=div.v)
    else:
        mean_div = util.Div()
    if normed:
        std_x = np.std(X, axis=0)
        X = X / std_x
        std_div = util.Div(d=std_x, i='standard deviation', v=div.v)
    else:
        std_div = util.Div()
    # PCA if N>P
    if n > p:
        T_T = np.dot(X.T, X)/n
        eigenvalues, eigenvectors = np.linalg.eig(T_T)
        idx_sort = (-eigenvalues).argsort()
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]
    # PCA if P>N
    else:
        TT_ = np.dot(X, X.T)/n
        eigenvalues, aux = np.linalg.eig(TT_)
        idx_sort = (-eigenvalues).argsort()
        eigenvalues = eigenvalues[idx_sort]
        aux = aux[:, idx_sort]
        eigenvec_not_normed = np.dot(aux.T, X).T
        norm = np.sqrt(np.sum(np.square(aux,aux), axis=1))
        eigenvectors = eigenvec_not_normed / norm
    
    # Avoid unstable representation
    for i in range(eigenvectors.shape[1]):
        test = np.max(np.abs(eigenvectors[:,i]))
        if test < 0:
            eigenvectors[:,i] = -eigenvectors[:,i]
    # Compute scores
    scores = np.dot(X, eigenvectors) / np.sqrt(n)
    variable_scores = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))
    
    # Compute explained variance
    variance_explained = np.round(eigenvalues/sum(eigenvalues)*100, 2)
    axisname = ['A' + str(i+1) + ' ' + str(variance_explained[i]) + '%' for i in range(variance_explained.shape[0])]
    
    # Outputs
    scores_div = util.Div(d=scores, i=div.i, v=axisname)
    eigenvec_div = util.Div(d=eigenvectors, i=axisname, v=div.v) # OK pour i et v ?
    loading_div = util.Div(d=eigenvectors/np.sqrt(n), i=axisname, v=div.v) # OK pour i et v ?

    eigenval_div = util.Div(d=eigenvalues, i=axisname, v='Eigenvalue') # OK pour i et v ?
    varscores_div = util.Div(d=variable_scores, i=div.v, v=axisname) # OK pour i et v ?

    # Info dict
    info_obj = util.Foo(type='pca', normed=normed, centred=centred)

    # Generate output
    pca_obj = util.Foo(info=info_obj, scores=scores_div, eigenvec=eigenvec_div, eigenval=eigenval_div, varscores=varscores_div, mean_x=mean_div, std_x=std_div, loadings=loading_div)

    return pca_obj
