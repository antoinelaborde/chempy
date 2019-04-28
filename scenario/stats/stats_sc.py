# -*- coding: utf-8 -*-
"""
@author: ALA
Script for utils
"""

%load_ext autoreload
%autoreload 2
# Import chempy and numpy packages
import chempy as cp
import numpy as np


# Let's import some data
# First, import spectral data
X = cp.read2div('./data_set/X1.CSV')

# Then, import Y values
Y = cp.read2div('./data_set/Y1.CSV')

#
corX = cp.cormap(X,X)

covX = cp.covmap(X,X)

distX = cp.distance(X,X)