# -*- coding: utf-8 -*-
"""
@author: ALA
Script for display functions
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

# Use curve quickly to display X data
cp.curve(X)
# Only display some indexes of X
cp.curve(X, i=[1,2])
# Use grouping to get list of div and display them with different colors
grouping_obj = cp.grouping(X, [1,2])
# Display curve with different colors and specify the filters expression as legend label
cp.curve(grouping_obj.div_list, legend_label=grouping_obj.filter_list)

# Select y column for color curve figure
Ycolor = cp.selectcol(Y, [18])
cp.curve(X, ycolor = Ycolor, cmap='Greens')