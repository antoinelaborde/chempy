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
        object instance with method apply

    Returns
    -------
    div containing scores

    """
    scores_div = loading_obj.apply(div)
    return scores_div