# -*- coding: utf-8 -*-
"""

@author: ALA
"""

# Import all sub-packages
__all__ = ['pca','ridge_regression','apply']


#from .pls import pls
from .pls_regression import pls
from .ridge_regression import ridge_regression
from .apply import apply_model