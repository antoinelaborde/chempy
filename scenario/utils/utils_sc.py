# -*- coding: utf-8 -*-
"""
@author: ALA
Script for utils
"""

#%load_ext autoreload
#%autoreload 2
# Import chempy and numpy packages
import chempy as cp
import numpy as np



# Let's import some data
# First, import spectral data
X = cp.read2div('./data_set/X1.CSV')
# Then, import Y values
Y = cp.read2div('./data_set/Y1.CSV')


# You can check the field of the data using field function
att_list, meth_list = cp.field(X)
# att_list contains the attributes of the X instance: d, i, v, id and p
# meth_list is empty for div structure


# Select some columns/variable in Y
Yselect = cp.selectcol(Y, ['Poids-kg','Ethylène (nmole/h.kg'])
# Select some rows in X
Xselect = cp.selectrow(X, ['be1p04','be2p11'])

# Select columns/variable in Y using the index
Yselect = cp.selectcol(Y, [0,1])
# Select some rows in X using the index
Xselect = cp.selectrow(X, list(range(0,10)))



# Delete some columns in X
Xdelete = cp.deletecol(X, ['1010.6439','1072.3626'])
# Delete some rows in Y
Ydelete = cp.deleterow(Y, ['be1p04','be2p11'])

# Delete some columns in X using indexes
Xdelete = cp.deletecol(X, list(range(0,100)))
# Delete some rows in Y using indexes
Ydelete = cp.deleterow(Y, [12,49])



# Append some columns
XY = cp.appendcol(X, Y)
# Be careful to non authorized column append, check the error for the line below
cp.appendcol(Xselect, Y)

# Append some rows
Xselect_bis = cp.selectrow(X, ['381p27','go2p15'])
X_appendrow = cp.appendrow(Xselect, Xselect_bis)



# Use grouping function to separate a div into several div according to a filter
# Let's consider an example with X.
# We choose [0,1] as filter indexes. This means that we are going to use the first (0) and the second (1) position in the .i field of X to construct filter and separate div.
group_struct = cp.grouping(X, [0,1])

# group_struct is a structure with 4 fields (use: group_struct.field()):
# div_list is a list containing each div that has been separated
# div_group_index is a div that contains the group index for each row of X
# div_group_number is a div that contains the number of rows in each div group
# filter_list is a list containing the expression of the filters

# Let's have a look to group_filter:
# group_struct.filter_list => [{0: '3', 1: '7'}, {0: 'g', 1: 'o'}, {0: 'm', 1: 'o'},{0: '3', 1: '8'}, {0: 'b', 1: 'e'}, {0: 'b', 1: 'l'}]

# Let's translate the first filter expression : {0: '3', 1: '7'}
# This expression is designed to filter row id that have a '3' at the first position (0) and a '7' at the second position (1)




# Now let's see a special case with grouping
# Let's consider you have a .i field which is only a 2 characters string and you request a filter on the 3rd position of .i field
X_grouping = cp.utils.util.copy(X)
X_grouping.i[100] = 'go'
group_struct = cp.grouping(X_grouping_bug, [0,2])

# When checking group_struct.filter_list, you see a special filter appears in first position: {'limit_size': 2}
# This means that some rows cannot be filtered by what you ordered to grouping function. Consequently, these rows are filtered out in a special div for which the index limit size is given by the filter expression (2)



# Create your Div instance from numpy array
my_array = np.random.randn(100,30)
# You can create Div only by specifying the d field. If so, .i and .v fields are automatically filled with integers.
my_div = cp.Div(d = my_array)

# Don't do this ! 
my_div = cp.Div()
my_div.d = my_array
# Doing that, you are not using the init class method of Div properly and you loose the advantage of having your i and v field automatically filled.

# If you provide false i or v field
false_i = np.random.randn(80)
# i field should have 100 element to be compatible with my_array
my_div = cp.Div(d = my_array, i=false_i)
# If you do this, the init method of Div will correct it by adding numerical rows.

# If you use the wrong way, nothing happens and you will have troubles after
my_div = cp.Div()
my_div.d = my_array
my_div.i = false_i




# Copy a Div instance: use the copy function of chempy ! 
Xcopy = cp.copy(X)
# Don't use this:
Xcopy = X
# Doing that, if you modify Xcopy, X will be modified as well. 




# Min/Max functions
# Use min/max function to find min/max values in your div
# If you specify the field 'v', you are looking for the variable that has the minimal value (for each row). 
Xmin_v = cp.min_div(X, field = 'v')
# Xmin_v is a structure of 2 div (try Xmin_v.field())
# Xmin_v.val is a div with the minimal value for each row
# Xmin_v.arg is a div with the information about the variable corresponding to the minimal value reported in Xmin_v.val

# For row 0, the minimal value among the variable is 0.32
Xmin_v.val.d[0]
# This value is obtained for the 35th variable of the array which is the column '1483.1778'
Xmin_v.arg.d[0,:]

# You get a symmetric behaviour with field = 'i'
Xmin_i = cp.min_div(X, field = 'i')

# If you specify an empty str for field, the global minimal value will be found.
# Then the Xmin_.arg is a div structure than contains corresponding indexes of row and colum and their names
Xmin_ = cp.min_div(X)





# Check the div fields
# The function check_duplicate helps you to identify duplicates in the d, i, v fields.
# First, let's create a div structure with duplicates
X_row1 = cp.selectrow(X, [100])
X_row2 = cp.selectrow(X, [200])

X_dup = cp.appendrow(X, X_row1)
X_dup = cp.appendrow(X_dup, X_row2)

X_col1 = cp.selectcol(X_dup, [4])
X_col2 = cp.selectcol(X_dup, [30])

X_dup = cp.appendcol(X_dup, X_col1)
X_dup = cp.appendcol(X_dup, X_col2)

# Use the check_duplicate function on X_dup
out = cp.check_duplicate(X_dup)
# out is a structure, you can check the fields : 
# out.field() --> ['duplicate_i', 'duplicate_v', 'duplicate_d']

# duplicate_i gives you information about duplicates in the i field
# out.duplicate_i --> {'go1p30': [100, 500], 'go3p11': [200, 501]}
# This information tells you that row names 'go1p30' and 'go3p11' occurs 2 times in the i field of X_dup

# duplicate_v gives you information about duplicates in the v field
# out.duplicate_v --> {'1492.8213': [30, 293], '1542.9678': [4, 292]}
# This information tells you that col names '1492.8213' and '1542.9678' occurs 2 times in the v field of X_dup

# duplicate_d gives you information about duplicates in the d field
# out.duplicate_d --> {'col': [[4, 292], [30, 293]], 'row': [[200, 501], [100, 500]]}
# This information tells you that colunms 4 and 292 in d are identical as well as columns 30 and 293. It also tells you that rows 200 and 501 are identical as well as rows 100 and 500.