# -*- coding: utf-8 -*-
"""
Created on 19/10/2018

@author: ALA
"""

import numpy as np
import chempy.utils.util as util


"""
def apply_model(model_obj, x_div, y_div=None):
"""

def apply_model(model_obj, x_div, y_div=None):
    """
    apply the model_obj to data in x_div
    Parameters
    ----------
    model_obj: Foo structure (mandatory)
        model structure
    x_div: div structure (mandatory)
        div containing data
    y_div: div structure (optional, default=None)
        div reference values

    Returns
    -------
    
    Credits
    -------

    """
    # List containing the available model that can be applied
    # model_obj must be a Foo class with info field containing one of the model type in AVAILABLE_MODEL
    AVAILABLE_MODEL = ['ridge_regression']

    if type(model_obj).__name__ != 'Foo':
        raise ValueError('model_obj argument is not an instance of the Foo class')
    if 'info' not in model_obj.field():
        raise ValueError('info about model_obj is not find: the model_obj you provide is not a model.')
    if model_obj.info.type not in AVAILABLE_MODEL:
        raise ValueError(str(model_obj.info.type) + ' is unknown for model application.')

    
    X = x_div.d
    if y_div is not None:
        Y = y_div.d
    else:
        Y = None
    n, p = X.shape

    # Model application
    if model_obj.info.type == 'ridge_regression':
        Beta = model_obj.beta
        mean_y = model_obj.mean_y
        Predy = np.dot(X, Beta.d) + mean_y.d
        if Y is not None:
            Error = Predy - Y
            Rmset = np.sqrt(np.mean(np.square(Error),axis=0))
            R2t = []
            for i in range(Beta.v.shape[0]):
                y_corr = np.corrcoef(np.ravel(Y), Predy[:,i])
                R2t.append(np.square(y_corr[0,1]))
            R2t = np.array(R2t)
        # Outputs
        rmset_div = util.Div(d=Rmset, i='RMSET', v=Beta.v)
        r2t_div = util.Div(d=R2t, i='R2T', v=Beta.v)
        predy_div = util.Div(d=Predy, i=x_div.i, v=Beta.v)

        out_obj = util.Foo(predy=predy_div, rmset=rmset_div, r2t=r2t_div)

    
    return out_obj