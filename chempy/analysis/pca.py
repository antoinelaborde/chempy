# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:00:54 2015

@author: DOMI
Updated by ALA 2018/10/10
"""

import numpy as np
import chempy.utils.util as util
import chempy.utils.classes as classes


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
    Pca class instance
    
    Credits
    -------
    Dominique Bertrand: dataframe@free.fr
    Antoine Laborde: antoine.laborde@greentropism.com
    """

    pca_obj = Pca(normed = normed, centred = centred)
    pca_obj.fit(div)

    return pca_obj


class Pca(classes.Foo):

    def __init__(self, normed = False, centred = True):
        """
        Parameters
        ----------
        normed: bool (optional, default=False)
            norm data before PCA
        centred: bool (optional, default=True)
            center data before PCA
        """
        self.normed = normed
        self.centred = centred
    
    def fit(self, div):
        """
        performs PCA on X
        Parameters
        ----------
        div: div structure (mandatory)
            div containing data
        """

        X = div.d
        n, p = X.shape

        if self.centred:
            mean_x = np.mean(X, axis=0, keepdims=True)
            X = X - mean_x
            self.mean_div = util.Div(d=mean_x, i='mean', v=div.v, id='mean')
        else:
            self.mean_div = util.Div(d=np.zeros((1,p)), i='mean', v=div.v, id='mean')
        if self.normed:
            std_x = np.std(X, axis=0, keepdims=True)
            X = X / std_x
            self.std_div = util.Div(d=std_x, i='standard deviation', v=div.v, id='std')
        else:
            self.std_div = util.Div(d=np.ones((1,p)), i='standard deviation', v=div.v, id='std')
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

        self.scores_div = util.Div(d=scores, i=div.i, v=axisname, id='scores')
        self.eigenvec_div = util.Div(d=eigenvectors, i=div.v, v=axisname, id='eigenvectors')
        self.loading_div = util.Div(d=eigenvectors/np.sqrt(n), i=div.v, v=axisname, id='loadings')
        self.eigenval_div = util.Div(d=eigenvalues, i=axisname, v='Eigenvalue', id='eigenvalues')
        self.varscores_div = util.Div(d=variable_scores, i=div.v, v=axisname, id='variables scores')
        self.variance_explained = variance_explained
    
    def apply(self, div):
        """
        apply the PCA object on X
        Parameters
        ----------
        div: div structure (mandatory)
            div containing data
        """
        X = div.d

        if self.centred:
            X = X - self.mean_div.d
        if self.normed:
            X = X / self.std_div.d
        scores = np.dot(X, self.loading_div.d)
        scores_div = util.Div(d=scores, i=div.i, v=self.scores_div.v)
        return scores_div