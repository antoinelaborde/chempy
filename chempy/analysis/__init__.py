# -*- coding: utf-8 -*-
"""

@author: ALA
"""

# Import all sub-packages
__all__ = ['pca','fda']

# Import all function to have a direct access like this:
# analysis.pca

from .pca import pca
from .fda import fda
from .factorial import compute_score

from .analysis import kruswal, anavar1, cormap, covmap, distance