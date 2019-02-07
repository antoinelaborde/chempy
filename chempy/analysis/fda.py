# -*- coding: utf-8 -*-
"""
Created on 28/01/2019

@author: ALA (from DOMI SAISIR Matlab)
"""

import numpy as np
import chempy.utils.util as util
import chempy.utils.classes as classes
from .pca import pca
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


"""
def fda(div, group, pca_dim, dim):
"""

def fda(div, group, pca_dim, dim, pca = True):
    """
    perform principal component analysis (PCA) on a div structure
    Parameters
    ----------
    div: div structure (mandatory)
        div containing data
    group: div structure (mandatory)
        div containing group indexes
    pca_dim : int
        fda performed on the pca_dim first dimensions of PCA
    dim: int
        maximum of factors for FDA
    pca: bool
        if True perform PCA before FDA

    Returns
    -------

    
    Credits
    -------
    Dominique Bertrand: dataframe@free.fr
    Reference: Bertrand et al., J of Chemometrics, Vol . 4, 413-427 (1990).

    Antoine Laborde: antoine.laborde@greentropism.com
    """

    fda_obj = Fda(pca_dim, dim, pca)
    fda_obj.fit(div, group)

    return fda_obj

class Fda(classes.Foo):
    
    def __init__(self, pca_dim, dim, pca):
        """
        Parameters
        ----------
        pca_dim : int
            fda performed on the pca_dim first dimensions of PCA
        dim: int
            maximum of factors for FDA
        """

        if dim >= pca_dim:
            dim = pca_dim - 1

        self.pca_bool = True
        self.pca_dim = pca_dim
        self.dim = dim

    def fit(self, div, group):
        """
        performs PCA then FDA on X
        Parameters
        ----------
        div: div structure (mandatory)
            div containing data
        group: div structure (mandatory)
            div containing group indexes
        """ 
        # Encode the classes
        le = LabelEncoder()
        le.fit(np.ravel(group.d))
        Y = le.transform(np.ravel(group.d))
        if self.pca_bool:
            # Perform PCA
            pca_obj = pca(div)
            X = pca_obj.scores_div
            X = util.selectcol(X,list(np.arange(self.pca_dim)))
        else:
            raise ValueError('Not implemented yet.')
        Xd = X.d
        n, _ = Xd.shape
        # Norm X columns
        Xd /= np.linalg.norm(Xd, axis=0)
        # Matrix classif
        mat_classif = util.binary_classif_matrix(classes.Div(d=Y))
        # Count class
        w = np.sum(mat_classif.d,axis=0)
        # Calculate gravity center of classes on PCA scores
        P = np.dot(np.dot(np.diag(1/w), mat_classif.d.T),Xd)

        # Criterion for the choice of score
        criterion_choice = np.diag(np.dot(np.dot(P.T,np.diag(w)),P))
        rank = np.argsort(-criterion_choice)
        
        confusion_mat = []
        accuracy_list = []
        variable_index_used = []
        group_hat = np.zeros((n,self.dim))
        for ind, i in enumerate(rank[:self.dim]):
            variable_index_used.append(i)                
                
            Pi = P[:,variable_index_used]
            Xi = Xd[:, variable_index_used]

            Pdiv = classes.Div(d=Pi)
            pca_P_obj = pca(Pdiv)

            # Determine the number of dimensions to keep
            n_, p = Pdiv.d.shape
            pca_P_obj_eivengec = pca_P_obj.eigenvec_div.d[:,:np.min(np.array([n_-1, p]))]

            # Data scores
            S = np.dot(Xi, pca_P_obj_eivengec)
            # Gravity center scores
            J = np.dot(Pi, pca_P_obj_eivengec)
            
            prediction = np.zeros(n)
            for j in range(n):
                delta = J - np.tile(S[j,:],(le.classes_.shape[0],1))
                dist = np.sum(np.multiply(delta,delta),axis=1)
                prediction[j] = np.argmin(dist)
            group_hat[:,ind] = prediction

            C = confusion_matrix(Y, prediction)
            Cdiv = classes.Div(d=C, i=le.classes_, v=le.classes_, id='Training Confusion Matrix FDA')
            confusion_mat.append(Cdiv)

            accuracy = np.trace(C)/np.sum(C)
            accuracy_list.append(accuracy)
            
        temp = np.diag(1/np.sqrt(pca_obj.eigenval_div.d[variable_index_used])[:,0])
        V = np.dot(np.dot(pca_obj.eigenvec_div.d[:,variable_index_used],temp), pca_P_obj_eivengec)
        beta = V/np.sqrt(n)
        self.beta = classes.Div(d=beta, id='Beta FDA')
        self.train_confusion = confusion_mat
        self.prediction = classes.Div(d=group_hat, v=np.arange(1,self.dim+1), i=div.i, id='Training Prediction FDA')
        self.mean_div = pca_obj.mean_div
        self.centroid_factor = classes.Div(d=J, id='Centroid factors FDA')
        self.train_accuracy = classes.Div(d=np.array(accuracy_list), i=np.arange(1,self.dim+1), id='Accuracy FDA')

    

    def apply(self, div, group=None):
        """
        performs PCA then FDA on X
        Parameters
        ----------
        div: div structure (mandatory)
            div containing data
        group: div structure (optional)
            div containing group indexes

        Returns
        -------
        prediction: div structure
            contains class prediction for the model with self.dim factors
        confusion_matrix: div structure (optional)
            confusion matrix if group is not None
        """
          
        Xd = X.d
        n, _ = Xd.shape
        # Center data
        Xd -= self.mean_div.d
        S = np.dot(Xd, self.beta)

        prediction = np.zeros(n)
        for j in range(n):
            delta = self.centroid_factor.d - S[j,:]
            dist = np.sum(np.multiply(delta,delta),axis=1)
            prediction[j] = np.argmin(dist)

        prediction = classes.Div(d=prediction, i=div.i)
        
        if group is not None:
            C = confusion_matrix(group.d, prediction)
            Cdiv = classes.Div(d=C, i=np.unique(group.d), v=np.unique(group.d))
            return prediction, Cdiv
        else:
            return prediction