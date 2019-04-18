# -*- coding: utf-8 -*-
"""

@author: ALA
"""

# Import all sub-packages
__all__ = ['analysis','plot','process','utils']

import matplotlib as mpl
mpl.use('Qt5Agg')

# Import all subpackage
from . import analysis
from . import plot
from . import process
from . import utils

# Import Div class
# Note: Foo is not available from chempy directly
from .utils.classes import  Div

# Import import function
from .utils.import_  import read2div, saisir2div, fileread



# Import function to get a direct access like this
# import chempy as cp


from .utils.import_ import read2div
from .utils.util import field, selectcol, selectrow, deletecol, deleterow, appendcol, appendrow, savediv, grouping, copy, transpose, col_index, row_index, min_div, max_div, mean_div, sum_div, check_duplicate

# cp.pca, cp.compute_score
from .analysis.pca import pca
from .analysis.fda import fda
from .analysis.factorial import compute_score

from .model.pls_regression import pls
from .model.ridge_regression import ridge_regression


from .model.apply import apply_model
from .plot.figure import curve, map2

from .process.preprocessing import process