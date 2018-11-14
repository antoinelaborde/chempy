# -*- coding: utf-8 -*-
"""
Created on 19/10/2018

@author: ALA

Functions for factorial analysis methods
"""

def compute_score(div, loading_obj):
    """
    computes score from a div structure and an object containing loadings
    Parameters
    ----------
    div: div structure (mandatory)
        div containing data
    loading_obj: object (mandatory)
        object containing the field 'loadings'

    Returns
    -------
    div containing scores

    """
    X = div.d
    # Pour le moment, je propose une méthode spécifique par loading_obj. A voir comment on peut homogeneiser la fonction avec le reste des loading_obj
    if loading_obj.info.type == 'pca':
        if loading_obj.info.centred:
            X = X - loading_obj.mean_x.d
        if loading_obj.info.normed:
            X = X / loading_obj.std_x.d
        scores = np.dot(X, loading_obj.loadings.d)

        scores_div = util.Div(d=scores, i=div.i, v=loading_obj.scores.v)
    
    return scores_div
