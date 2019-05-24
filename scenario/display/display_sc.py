# -*- coding: utf-8 -*-
"""
@author: ALA
Script for display functions
"""

# %load_ext autoreload
# %autoreload 2
# Import chempy and numpy packages

import chempy as cp
import numpy as np
import chempy.utils.classes as classes
#isinstance(X.i,np.ndarray)
#Out[17]: True
#
#isinstance(X.v,np.ndarray)
#Out[18]: True

# Let's import some data
# First, import spectral data
X = cp.read2div('X1.CSV')
# Then, import Y values
Y = cp.read2div('Y1.CSV')

# Use curve quickly to display X data
print(type(X))
cp.curve(X)
# Only display some indexes of X
cp.curve(X, row=[1,2])
# Use grouping to get list of div and display them with different colors
grouping_obj = cp.grouping(X, [1,2])
# Display curve with different colors and specify the filters expression as legend label
cp.curve(grouping_obj.div_list, legend_label=grouping_obj.filter_list)

# Select y column for color curve figure
Ycolor = cp.selectcol(Y, [18])
cp.curve(X, ycolor = Ycolor, cmap='Greens')


# Let's test a PCA
pca_obj = cp.pca(X)
# Maps the scores : X axis is the first score, Y axis is the second score
cp.map2(pca_obj.scores_div,0 ,1)
# Only display indexes from 10 to 99
cp.map2(pca_obj.scores_div,0,1,row=list(range(10,100)))
# Add a colormap with respect to Y div
cp.map2(pca_obj.scores_div,0,1,ycolor=Ycolor, cmap='Greens')
# Use grouping
grouping_obj = cp.grouping(pca_obj.scores_div, [1,2])
cp.map2(grouping_obj.div_list,0,1,legend_label=grouping_obj.filter_list, cmap='Set1')
