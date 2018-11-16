# -*- coding: utf-8 -*-
"""
@author: ALA
Script for principal component analysis
"""

%load_ext autoreload
%autoreload 2
# Import chempy and numpy packages
import chempy as cp
import numpy as np

# Let's import some data
# First, import spectral data
X = cp.read2div('./scenario/pca/X_pca_sc.CSV')
# Then, import Y values
Y = cp.read2div('./scenario/pca/Y_pca_sc.CSV')


# Cut X in two parts
X1 = cp.selectrow(X, list(np.arange(0,200)))
X2 = cp.selectrow(X, list(np.arange(200,X.d.shape[0])))

# Let's apply PCA on X
pca_obj = cp.pca(X1)
# pca_obj is an instance of the class PCA.
# PCA inherits from Foo, so you can get object fields easily using:
# pca_obj.field()

# Apply PCA on X2
scores_div = cp.compute_score(X2, pca_obj)
