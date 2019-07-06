# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:59:17 2015

@author: DOMI, ALA
"""
# Import ######################################
import os
import csv
import numpy as np
from scipy.io import loadmat
import shelve
import datetime
import matplotlib.pyplot as pl
from copy import deepcopy
from .classes import Div, Foo

"""
field(test):
copy(div):
deletecol(div,filter):
deleterow(div,filter): 
selectcol(div,filter):
selectrow(div,filter):
appendcol(*div):
appendrow(*div):
savediv(div, filename, delimiter=';'): 
grouping(div, filter):
row_index(div, row):
col_index(div, col):
transpose(div):
row_index(div, row):
col_index(div, row):
min_div(div, field = ''):
max_div(div, field = ''):
mean_div(div, field = ''):
sum_div(div, field = ''):
check_duplicate(div):
isdiv(obj):
randomize(div)
reorder(div1,div2):
"""

def field(obj):
    """
    field returns the fields of an object's instance
    Parameters
    ----------
    obj: instance (mandatory)
        instance of a class containing fields
    Returns
    -------
    tuple of list

    attr_list, method_list = field(instance)
    attr_list = ['attribute1', 'attribute2',...]
    method_list = ['method1', 'method2',...]

    """

    method_list=[]
    for method_name in dir(obj):
        if method_name[0:2] != '__':
            if callable(getattr(obj, method_name)):
                method_list.append(method_name)

    attr_list = list(obj.__dict__.keys())
    return attr_list, method_list

def copy(div):
    """
    deep copy of div instance
    Parameter
    ---------
    div: instance of div
    Return
    a copy of div
    """
    return deepcopy(div)
    
def deletecol(div, col_filter):
    """
    deletes column in a div instance
    Parameters
    ----------
    col_filter: list of str or int
        name of the column to delete or index of column to delete
    Returns
    ----------
    filt_div: div instance with only filtered columns

    """
    if not(isinstance(col_filter, list)):
        raise ValueError('col_filter must be a list')

    # Check the type of filter: col names or col index
    if all([isinstance(filt, str) for filt in col_filter]):
        filter_case = 'str'
    elif all([isinstance(filt, (np.integer, int)) for filt in col_filter]):
        filter_case = 'index'
    else:
        raise ValueError('col_filter argument must contain only str or int, not both.')

    if filter_case == 'str':
        filt_index = []
        for filt in col_filter:
            if filt not in div.v:
                raise ValueError(filt + ' is not in div.v')
            else:
                index = np.where(div.v == filt)[0].ravel()[0]
                filt_index.append(index)
    elif filter_case == 'index':
        # Check index are in the range
        if max(col_filter) >= div.v.shape[0] or max(col_filter) < 0:
            raise ValueError('index in col_filter must be in the range [0, ' + str(div.v.shape[0]) + ']')
        else:
            filt_index = col_filter
    
    filt_div = copy(div)
    filt_div.d = np.delete(div.d, filt_index, axis=1)
    filt_div.v = np.delete(div.v, filt_index)
    filt_div.id = filt_div.id +' deletecol using filter ' + str(col_filter)
    
    return filt_div

def deleterow(div, row_filter):
    """
    deletes rows in a div instance
    Parameters
    ----------
    row_filter: list of str or int
        name of the row to select or index of row to select
    Returns
    ----------
    filt_div: div instance with only filtered rows

    """
    if not(isinstance(row_filter, list)):
        raise ValueError('row_filter must be a list')

    # Check the type of filter: row names or row index
    if all([isinstance(filt, str) for filt in row_filter]):
        filter_case = 'str'
    elif all([isinstance(filt, (np.integer, int)) for filt in row_filter]):
        filter_case = 'index'
    else:
        raise ValueError('row_filter argument must contain only str or int, not both.')
    
    if filter_case == 'str':
        filt_index = []
        for filt in row_filter:
            if filt not in div.i:
                raise ValueError(filt + ' is not in div.i')
            else:
                index = np.where(div.i == filt)[0].ravel()[0]
                filt_index.append(index)
    elif filter_case == 'index':
        # Check index are in the range
        if max(row_filter) >= div.i.shape[0] or max(row_filter) < 0:
            raise ValueError('index in row_filter must be in the range [0, ' + str(div.i.shape[0]) + ']')
        else:
            filt_index = row_filter
    
    filt_div = copy(div)
    filt_div.d = np.delete(div.d, filt_index, axis=0)
    filt_div.i = np.delete(div.i, filt_index)
    filt_div.id = filt_div.id +' deleterow using filter ' + str(row_filter)
    
    return filt_div
    
def selectcol(div, col_filter):    
    """
    selects column in a div instance
    Parameters
    ----------
    col_filter: list of str or int
        name of the column to select or index of column to select

    Returns
    ----------
    filt_div: div instance with only filtered columns

    """
    if not(isinstance(col_filter, list)):
        raise ValueError('col_filter must be a list')

    # Check the type of filter: col names or col index
    if all([isinstance(filt, str) for filt in col_filter]):
        filter_case = 'str'
    elif all([isinstance(filt, (np.integer, int)) for filt in col_filter]):
        filter_case = 'index'
    else:
        raise ValueError('col_filter argument must contain only str or int, not both.')
    
    if filter_case == 'str':
        filt_index = []
        for filt in col_filter:
            if filt not in div.v:
                raise ValueError(str(filt) + ' is not in div.v')
            else:
                index = np.where(div.v == filt)[0].ravel()[0]
                filt_index.append(index)
    elif filter_case == 'index':
        # Check index are in the range
        if max(col_filter) >= div.v.shape[0] or max(col_filter) < 0:
            raise ValueError('index in col_filter must be in the range [0, ' + str(div.v.shape[0]) + ']')
        else:
            filt_index = col_filter
            
    filt_div = copy(div)
    filt_div.d = div.d[:, filt_index]
    filt_div.v = div.v[filt_index]
    filt_div.id = filt_div.id +' selectcol using filter ' + str(col_filter)
    
    return filt_div

def selectrow(div, row_filter):    
    """
    selects rows in a div instance
    Parameters
    ----------
    row_filter: list of str or int
        name of the row to select or index of row to select
    Returns
    ----------
    filt_div: div instance with only filtered rows

    """
    if not(isinstance(row_filter, list)):
        raise ValueError('row_filter must be a list')
    
    # Check the type of filter: row names or row index
    if all([isinstance(filt, str) for filt in row_filter]):
        filter_case = 'str'
    elif all([isinstance(filt, (np.integer, int)) for filt in row_filter]):
        filter_case = 'index'
    else:
        raise ValueError('row_filter argument must contain only str or int, not both.')
    
    if filter_case == 'str':
        filt_index = []
        for filt in row_filter:
            if filt not in div.i:
                raise ValueError(str(filt) + ' is not in div.i')
            else:
                index = np.where(div.i == filt)[0].ravel()[0]
                filt_index.append(index)
    elif filter_case == 'index':
        # Check index are in the range
        if max(row_filter) >= div.i.shape[0] or max(row_filter) < 0:
            raise ValueError('index in row_filter must be in the range [0, ' + str(div.i.shape[0]) + ']')
        else:
            filt_index = row_filter
    
    filt_div = copy(div)
    filt_div.d = div.d[filt_index, :]
    filt_div.i = div.i[filt_index]
    filt_div.id = filt_div.id +' selectrow using filter ' + str(row_filter)
    
    return filt_div

def appendcol(div_list):
    """
    add columns of Div instances
    Parameters
    ----------
    div: list
        list of div instances
    Return
    ------
    div instance

    """
    # Check compatibility
    for ind, div_ in enumerate(div_list):
        if ind == 0:
            ref_n = div_.d.shape[0]
            ref_i = div_.i
        else:
            test_n = div_.d.shape[0]
            test_i = div_.i
            if test_n != ref_n:
                raise ValueError('div#' + str(ind) + ' does not have the good number of rows. div#' + str(ind) + ' has ' + str(test_n) + ' rows but the reference div has ' + str(ref_n) + ' rows')
            if not(np.array_equal(test_i, ref_i)):
                print('Warning: div#' + str(ind) + '.i is different from the reference div.i')

    for ind, div_ in enumerate(div_list):
        if ind == 0:
            cumul_div = copy(div_)
        else:
            cumul_div.d = np.append(cumul_div.d, div_.d, axis=1)
            cumul_div.v = np.append(cumul_div.v, div_.v, axis=0)
    
            cumul_div.id = cumul_div.id + div_.id
    
    return cumul_div
    
def appendrow(div_list):
    """
    add rows of Div instances
    Parameters
    ----------
    div: list
        list of div instances
    Return
    ------
    div instance

    """
    # Check compatibility
    for ind, div_ in enumerate(div_list):
        if ind == 0:
            ref_p = div_.d.shape[1]
            ref_v = div_.v
        else:
            test_p = div_.d.shape[1]
            test_v = div_.v
            if test_p != ref_p:
                raise ValueError('div#' + str(ind) + ' does not have the good number of columns. div#' + str(ind) + ' has ' + str(test_p) + ' columns but the reference div has ' + str(ref_p) + 'columns')
            if not(np.array_equal(test_v, ref_v)):
                print('Warning: div#' + str(ind) + '.v is different from the reference div.v')

    for ind, div_ in enumerate(div_list):
        if ind == 0:
            cumul_div = copy(div_)
        else:
            cumul_div.d = np.append(cumul_div.d, div_.d, axis=0)
            cumul_div.i = np.append(cumul_div.i, div_.i, axis=0)
    
            cumul_div.id = cumul_div.id + div_.id
    
    return cumul_div
 
def savediv(div, filename, delimiter=';'):
    """
    savediv saves a div structure in a csv file
    Parameters
    ----------
    div: div instance

    filename: str
        name of the csv file

    Return
    ------

    """   
    with open(filename,'w') as f:
        aux = delimiter.join(str(n) for n in div.v)
        aux = ' '+ delimiter + aux + '\n' #necessary to ad a void cell for excel
        f.write(aux)
        for i in range(0,div.i.shape[0]):
            aux = delimiter.join(str(n) for n in div.d[i,:])    
            aux = div.i[i] + delimiter + aux + '\n'
            f.write(aux)

def grouping(div, filter):
    """
    grouping separate div into several div by filtering using the div.i field at the position specified by the filter argument

    Parameters
    ----------
    div: div instance

    filter: list of int
        list of positions in the div.i for considering filtering

    Notes
    -----
    Assume the div.i contains:
    '381p03'
    '381p07'
    '381p08'
    '331p03'
    '331p08'
    '383p08'
    If you use grouping(div, [1]), grouping will separate rows according to the 2nd position in the character string : 
    div1: ['381p03', '381p07', '381p08']
    div2: ['331p03', '331p08']
    div3: ['383p08']

    If you use grouping(div, [0,1]), grouping will sperate rows according to the first and the second position : 
    div1: ['381p03', '381p07', '381p08', '383p08']
    div2: ['331p03', '331p08']

    Return
    ------
    object with attributes:
        div_list: list of div
        div_group_index: div
        div_group_number: div

    Credits
    -------
    Dominique Bertrand
    Antoine Laborde

    """
    filter_limit_size = None
    # Find all the filter values
    # List contains information for filtering like dict:
    # Ex: {0:'d', 2:'1'} => position 0 is 'd', position 2 is '1'
    filter_val_list = []

    # Loop on div.i rows
    for val in div.i:
        # Create a filter dict
        filter_dict = {}
        for filt_pos in filter:
            # Prevent index too high by creating a new filter for small .i fields
            if filt_pos >= len(val):
                filter_limit_size = {}
                filter_limit_size['limit_size'] = filt_pos
            else:
                filter_dict[filt_pos] = val[filt_pos]
        # Put filter_dict in filter_val_list if it's new
        if filter_dict not in filter_val_list:
            filter_val_list.append(filter_dict)

    # Warning: if limit_size filter is in filter_val_list, it must be in the first place
    if filter_limit_size is not None:
        filter_val_list = [filter_limit_size] + filter_val_list

    index_match_filter = []
    # Loop on index
    for ind, identifier in enumerate(div.i):
        # Loop on filters
        for filt_ind, filt in enumerate(filter_val_list):
            filter_ok = True
            # Loop on filter conditions
            for filt_key in filt.keys():
                # Special filter for checking the size anomaly
                if filt_key == 'limit_size':
                    if len(identifier) > filt[filt_key]:
                        filter_ok = False
                elif identifier[filt_key] != filt[filt_key]:
                    filter_ok = False
            if filter_ok:
                break
        index_match_filter.append(filt_ind)
    index_match_filter = np.array(index_match_filter)
    group_number = []
    # Filter div
    div_list = []
    filter_id = []
    filter_list = []
    for index in np.unique(index_match_filter):
        select_filter_div = np.where(index_match_filter == index)[0].ravel().tolist()
        div2out = selectrow(div, select_filter_div)
        div_list.append(div2out)
        group_number.append(div2out.d.shape[0])
        filter_id.append(str(filter_val_list[index]))
        filter_list.append(filter_val_list[index])

    group_number = np.array(group_number)
    filter_id = np.array(filter_id)
    # Create div structure for div_group_index and div_group_number
    var_name = np.array(['group index for filter ' + str(filter)])
    div_group_index = Div(d = index_match_filter, i = div.i, v = var_name)
    div_group_number = Div(d = group_number, i = filter_id, v = 'group cardinal')

    # Create Foo structure for output
    group_obj = Foo(div_list = div_list, div_group_index = div_group_index, div_group_number = div_group_number, filter_list = filter_list)

    return group_obj
  
def transpose(div):
   """
   Transpose of a Div class 
   Parameters
   ----------
   div: an instance of Div class   
   return
   ----------   
   div transposed (field ".d" transposed, div.i=div.v, div.=div.i)
   """   
   
   div_t = Div(d = np.transpose(div.d), i = div.v, v = div.i, id = div.id+' transposed')
   return div_t

def row_index(div, row):
    """
    row_index returns the row index corresponding for which row == div.i
    Parameters
    ----------
    div: an instance of Div class   
    return
    ----------   
    list
    """
    row_index = np.where(div.i == row)[0].ravel().tolist()
    return row_index
    
def col_index(div, col):
    """
    col_index returns the col index corresponding for which col == div.v
    Parameters
    ----------
    div: an instance of Div class   
    return
    ----------   
    list
    """
    col_index = np.where(div.v == col)[0].ravel().tolist()
    return col_index

def min_div(div, field = ''):
    """
    Return min value of a div
    Parameters
    ----------
    div: an instance of Div class

    field: str
        'v' or 'i' or ''
    
    Notes
    -----
    if field == 'v':
        calculates min value for among the variable for each row
    if field == 'i':
        calculates min value for among the variable for each row
    if field == '':
        min calculates the global minimum value

    
    return
    ------
    structure with fields:
        val: div with min values
        arg: argument of min values

    """

    if field == 'v':
        d_min_val = np.min(div.d, axis=1)
        d_min_ind = np.argmin(div.d, axis=1)
        v_min = div.v[d_min_ind]

        d_min_field = np.array([d_min_val]).T
        d_argmin_field = np.array([v_min, d_min_ind]).T

        i_field = div.i
        v_minfield = np.array(['min value'])
        v_argminfield = np.array(['min col name','min col index'])
        
        div_min = Div(d = d_min_field, i = i_field, v = v_minfield, id = div.id + ' min v')

        div_argmin = Div(d = d_argmin_field, i = i_field, v = v_argminfield, id = div.id + ' argmin v')
    
    elif field == 'i':
        d_min_val = np.min(div.d, axis=0)
        d_min_ind = np.argmin(div.d, axis=0)
        i_min = div.i[d_min_ind]

        d_min_field = np.array([d_min_val])
        d_argmin_field = np.array([i_min, d_min_ind])

        v_field = div.v
        i_minfield = np.array(['min value'])
        i_argminfield = np.array(['min row name', 'min row index'])

        div_min = Div(d = d_min_field, i = i_minfield, v = v_field, id = div.id + ' min i')

        div_argmin = Div(d = d_argmin_field, i = i_argminfield, v = v_field, id = div.id + ' argmin i')
    
    elif field == '':
        d_min_val = np.min(div.d)
        d_min_ind = np.unravel_index(div.d.argmin(), div.d.shape)
        div_min = selectrow(div, [d_min_ind[0]])
        div_min = selectcol(div_min, [d_min_ind[1]])

        argmin = np.array([np.array(d_min_ind), np.array([div.i[d_min_ind[0]], div.v[d_min_ind[1]]])])

        div_argmin = Div(d = argmin.T, i = np.array(['row','col']), v = ['min name','min index'], id = div.id + ' argmin')


    out_obj = Foo(val = div_min, arg = div_argmin)
            
    return out_obj
    
def max_div(div, field = ''):
    """
    Return max value of a div
    Parameters
    ----------
    div: an instance of Div class

    field: str
        'v' or 'i' or ''
    
    Notes
    -----
    if field == 'v':
        calculates max value for among the variable for each row
    if field == 'i':
        calculates max value for among the row for each variable
    if field == '':
        max calculates the global maximum value

    
    return
    ------
    structure with fields:
        val: div with max values
        arg: argument of max values

    """

    if field == 'v':
        d_max_val = np.max(div.d, axis=1)
        d_max_ind = np.argmax(div.d, axis=1)
        v_max = div.v[d_max_ind]

        d_max_field = np.array([d_max_val]).T
        d_argmax_field = np.array([v_max, d_max_ind]).T

        i_field = div.i
        v_maxfield = np.array(['max value'])
        v_argmaxfield = np.array(['max col name','max col index'])
        
        div_max = Div(d = d_max_field, i = i_field, v = v_maxfield, id = div.id + ' max v')

        div_argmax = Div(d = d_argmax_field, i = i_field, v = v_argmaxfield, id = div.id + ' argmax v')
    
    elif field == 'i':
        d_max_val = np.max(div.d, axis=0)
        d_max_ind = np.argmax(div.d, axis=0)
        i_max = div.i[d_max_ind]

        d_max_field = np.array([d_max_val])
        d_argmax_field = np.array([i_max, d_max_ind])

        v_field = div.v
        i_maxfield = np.array(['max value'])
        i_argmaxfield = np.array(['max row name', 'max row index'])

        div_max = Div(d = d_max_field, i = i_maxfield, v = v_field, id = div.id + ' max i')

        div_argmax = Div(d = d_argmax_field, i = i_argmaxfield, v = v_field, id = div.id + ' argmax i')
    
    elif field == '':
        d_max_val = np.max(div.d)
        d_max_ind = np.unravel_index(div.d.argmax(), div.d.shape)
        div_max = selectrow(div, [d_max_ind[0]])
        div_max = selectcol(div_max, [d_max_ind[1]])

        argmax = np.array([np.array(d_max_ind), np.array([div.i[d_max_ind[0]], div.v[d_max_ind[1]]])])

        div_argmax = Div(d = argmax.T, i = np.array(['row','col']), v = ['max name','max index'], id = div.id + ' argmax')


    out_obj = Foo(val = div_max, arg = div_argmax)
            
    return out_obj

def mean_div(div, field = ''):
    """
    Return the mean of d according to the field
    Parameters
    ----------
    div: an instance of Div class

    field: str
        'v' or 'i' or ''
    
    Notes
    -----
    if field == 'v':
        calculates the mean among the variables for each row
    if field == 'i':
        calculates the mean among the rows for each variable
    if field == '':
        calculates the global mean value

    
    return
    ------
    div

    """

    if field == 'v':
        d_mean = np.mean(div.d, axis=1, keepdims=1)

        i_field = div.i
        v_field = np.array(['mean value'])
        id_add = ' mean v'
            
    elif field == 'i':
        d_mean = np.mean(div.d, axis=0, keepdims=1)

        v_field = div.v
        i_field = np.array(['mean value'])
        id_add = ' mean i'
            
    elif field == '':
        d_mean = np.mean(div.d, keepdims=1)

        v_field = np.array(['mean value'])
        i_field = np.array(['mean value'])
        id_add = ' mean'
        
    div_mean = Div(d = d_mean, i = i_field, v = v_field, id = div.id + id_add)

    return div_mean

def group_mean(div,group):
    """
    Compute the means according to a grouping
    Parameters
    ----------
    div: data to be averaged by group
    group: integer giving the group number for each row
    Return
    ------
    center: div of the averages by group
    group_size: number of observations belonging to the corresponding group

    """
    maxgroup=int(max(group.d))
    xcenter=np.zeros((maxgroup,div.d.shape[1]))
    #np.zeros(3, dtype = int)
    aux=np.zeros(maxgroup,dtype=int)
    for i in range(1,maxgroup+1):
        index_group=np.where(group.d == i)[0]
        #print(index_group.shape)
        #print(i,div.d[index_group,:].shape[0])
        aux[i-1]=np.asarray(div.d[index_group,:].shape)[0]       
        xcenter[i-1,:]=np.mean(div.d[index_group,:],axis=0)
    center=Div(d=xcenter,i=np.array(list(range(1,(maxgroup+1)))),v=div.v) 
    group_size=Div(aux,i=np.array(list(range(1,maxgroup+1))),v='group size')
    return center,group_size
    #print(np.sum(aux))   
 
def sum_div(div, field = ''):
    """
    Return the sum of d according to the field
    Parameters
    ----------
    div: an instance of Div class

    field: str
        'v' or 'i' or ''
    
    Notes
    -----
    if field == 'v':
        calculates sum among the variable for each row
    if field == 'i':
        calculates sum among the row for each variable
    if field == '':
        calculates the global sum value

    
    return
    ------
    div

    """

    if field == 'v':
        d_sum = np.sum(div.d, axis=1, keepdims=1)

        i_field = div.i
        v_field = np.array(['sum value'])
        id_add = ' sum v'
            
    elif field == 'i':
        d_sum = np.mean(div.d, axis=0, keepdims=1)

        v_field = div.v
        i_field = np.array(['sum value'])
        id_add = ' sum i'
            
    elif field == '':
        d_sum = np.mean(div.d, keepdims=1)

        v_field = np.array(['sum value'])
        i_field = np.array(['sum value'])
        id_add = ' sum'
        
    div_sum = Div(d = d_sum, i = i_field, v = v_field, id = div.id + id_add)

    return div_sum
    
def check_duplicate(div):
    """
    Return list of index that are duplicated in your d,i,v fields
    Parameters
    ----------
    div: an instance of Div class
    
    Notes
    -----
    1 - check that every row name is different. If not, send back index of row that have identical row names

    2 - check that every col name is different. If not, send back index of col that have identical col names

    3 - check that every line and every col are different. If not, send back index of col/ind that are identical

    
    return
    ------
    structure with fields:
    - duplicate_i: dict with row .i as keys and row index of duplicates as val
    - duplicate_v: dict with col .v as keys and col index of duplicates as val
    - duplicate_d: dict with 2 fields 'row' and 'col', values are lists grouping index that are identical row and col of .d field of the div

    """

    # 1 - Check i field
    unique_i = np.unique(div.i)
    i_duplicate_dict = None
    if unique_i.shape[0] != div.i.shape[0]:
        i_duplicate_dict = {}
        for i_row in unique_i:
            row_index = []
            for ind_i, ii_row in enumerate(div.i):
                if i_row == ii_row:
                    row_index.append(ind_i)
            if len(row_index) > 1:
                i_duplicate_dict[i_row] = row_index

    # 2 - Check i field
    unique_v = np.unique(div.v)
    v_duplicate_dict = None
    if unique_v.shape[0] != div.v.shape[0]:
        v_duplicate_dict = {}
        for v_col in unique_v:
            col_index = []
            for ind_v, vv_col in enumerate(div.v):
                if v_col == vv_col:
                    col_index.append(ind_v)
            if len(col_index) > 1:
                v_duplicate_dict[v_col] = col_index
    
    # 3 - Check d field
    unique_drow = np.vstack({tuple(row) for row in div.d})
    unique_dcol = np.vstack({tuple(col) for col in div.d.T}).T
    d_duplicate_dict = {'row':[], 'col':[]}
    if unique_drow.shape[0] != div.d.shape[0]:
        for row in unique_drow:
            row_index = []            
            for ind_r, drow in enumerate(div.d):
                if np.array_equal(row, drow):
                    row_index.append(ind_r)
            if len(row_index) > 1:
                d_duplicate_dict['row'].append(row_index)
    if unique_dcol.shape[1] != div.d.shape[1]:
        for col in unique_dcol.T:
            col_index = []
            for ind_c, dcol in enumerate(div.d.T):
                if np.array_equal(col, dcol):
                    col_index.append(ind_c)
            if len(col_index) > 1:
                d_duplicate_dict['col'].append(col_index)

    out_obj = Foo(duplicate_i = i_duplicate_dict, duplicate_v = v_duplicate_dict, duplicate_d = d_duplicate_dict)

    return out_obj

def vfield2num(div):
    """
    vfield2num tests if div.v can be interpreted as numericals
    Parameters
    ----------
    div: an instance of Div class
    
    return
    ------
    None if vfield cannot be interpreted as numericals
    numpy array if vfield can be interpreted as numericals

    """
    vfield = div.v
    try:
        vfield = vfield.astype('float')
    except ValueError:
        vfield = None
    return vfield

def isdiv(obj):
    """
    isdiv tests if obj is a div instance
    Parameters
    ----------
    obj: a python objet
    
    return
    -----
    true if obj has field d, i, v, else otherwise 

    Note
    ------
    The function only verifies that fields d, i, v exist in the object. This
    is sufficient in practice to give confidence that the object is an instance
    of class Div. This function tries to replace the function isinstance of Python
    wich is not correctly working in some situation.
    """
    return(hasattr(obj, 'd') and hasattr(obj, 'i') and hasattr(obj, 'v'))
    
def save_workspace(local_out):
    """
    Save all variables in the ipython workspace
    Parameters
    ----------
    local_out: result of the locals() python built-in function

    ----------   
    """
    # System variable to get out
    VARIABLE_OUT = ['exit','quit','In','Out','get_ipython']
    # Get list of all variables
    true_variable_list = []
    # For each variable name in local_out, decision_dict provides the decision of the test if the variable has to be stored or not
    decision_dict = {}

    for name in local_out.keys():
        inlist = True
        if name[:1] == '_':
            decision_dict[name] = 'Not defined by user'
            inlist = False
        if type(local_out[name]).__name__ == 'module':
            decision_dict[name] = 'Is a module'
            inlist = False
        if name in VARIABLE_OUT:
            decision_dict[name] = 'Not defined by user'
            inlist = False
        if inlist:
            true_variable_list.append(name)

    # Construct the dict to save
    save_dict = {}
    for var2save in true_variable_list:
        save_dict[var2save] = local_out[var2save]
    
    # Current time
    time_save = datetime.datetime.now()
    timestr = time_save.strftime('%Y%m%d_%H%M%S')
    
    filename_str = timestr +'_workspace'

    # Use Dump with shelve module
    d = shelve.open(filename_str)
    d['datastore'] = save_dict
    d.close()

    print('Workspace is saved in: ' + filename_str)

def load_workspace(filename, globalres):
    """
    Load all variables saved in a file using save_workspace function
    Parameters
    ----------
    filename: str
        name of the file
    globalres: result of the globals() python built-in

    ----------   
    """
    with shelve.open(filename) as f:
        var_dict = f['datastore']

    globalres.update(var_dict)


def binary_classif_matrix(group):
    """
    Create a matrix that describes to what class belongs each row of the group vector
    Parameters
    ----------
    group: div instance
        div discribing the group belongings

    Returns
    ----------
    group_mat: div instance
        for each row, is one for the corresponding class   
    """

    # Group must be a div with one dimension
    if 1 not in group.d.shape:
        raise ValueError('group must be a div with dimensions n x 1')
    
    if group.d.shape[1] != 1:
        group_val = group.d.T
    else:
        group_val = group.d

    # Number of groups
    unique_group = np.unique(group_val)
    n_group = unique_group.shape[0]
    n_row = group_val.shape[0]
    # Init classif matrix
    classif_matrix = np.zeros((n_row,n_group))

    for i in np.arange(n_row):
        group_index = np.ravel(np.where((unique_group == group_val[i])))
        classif_matrix[i,group_index] = 1

    classif_div = Div(d=classif_matrix, i=group.i, v=unique_group)
    return classif_div

def randomize(div):
     """
     return div with rows in random order
     Parameters
     ----------
     div: an instance of Div class
     
     return
     ------
     a div matrix with the order of the rows randomized
     
     """
     #print(type(div))
     if(not(isdiv(div))):
         raise ValueError('the entered argument is not an instance of class div')

     rorder=np.random.permutation(np.arange(div.i.shape[0]))
     #print(type(rorder))
     return(selectrow(div,rorder.tolist()))

def reorder(div1,div2):
    """
    This function makes it possible to realign the rows of div1 and div2, in order
    to have the identifiers corresponding.
    The function discards the observations which are not present in d1v and div2.
    Fails if div1 or div2 contains duplicate (identical) identifiers of rows. 
    The rows in the resulting out Div instances are sorted in the alphabetic order of 
    the rows identifiers (.i)
    
    Parameters
    ----------
    div1, div2 instances of class Div to be reordered

    Returns
    ----------
    out1, out2 : reordered Div instances corresponding to div1 and div2 respectively
    diff1, diff2 : list of rows names with no correspondance found in div1 and div2 repsectively 
    """
    if(not(isdiv(div1))):
            raise ValueError('the entered first argument is not an instance of class div')
    if(not(isdiv(div2))):
            raise ValueError('the entered second argument is not an instance of class div')
    if((np.shape(div1.i)[0])!= (np.shape(np.unique(div1.i))[0])):
        raise ValueError('the first argument has not unique row identifiers. Impossible to reororder')
    #np.intersect1d(X.i, rX.i
    if((np.shape(div2.i)[0])!= (np.shape(np.unique(div2.i))[0])):
        raise ValueError('the second argument has not unique row identifiers. Impossible to reororder')
    common=np.intersect1d(div1.i,div2.i)
    list1=list()
    list2=list()
    for index, thisname in enumerate(common):
        row_index1 = np.where(div1.i == thisname)[0]#.ravel().tolist()
        list1.append(row_index1[0])
        row_index2 = np.where(div2.i == thisname)[0]#.ravel().tolist()
        list2.append(row_index2[0])
    out1=selectrow(div1,list1)
    out2=selectrow(div2,list2)
    diff1=list(set(div1.i).difference(set(common))) 
    diff2=list(set(div2.i).difference(set(common))) 
    
    return(out1,out2,diff1,diff2)

def quantif_perf(y, yh, nb_variables=None):
    """
    calculate performance indicators for regression results
    Parameters
    ----------
    y: div structure (mandatory)
        div reference values
    yh: div structure (mandatory)
        div predicted values
    nb_variables: int (optional, default=None)
        number of variables used
    """

    # RMSE
    rmse = np.sqrt(np.sum(np.square(y - yh)))
    # R2
    ssres = np.sum((yh - y)**2)
    sstot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ssres/sstot

    out = {'rmse':rmse, 'r2':R2}
    if nb_variables is not None:
        n = y.shape[0]
        # BIC
        BIC = n*np.log(ssres/n) + nb_variables*np.log(n)
        # AIC
        AIC = n*np.log(ssres/n) + 2*nb_variables
        out['AIC'] = AIC
        out['BIC'] = BIC

    return out

def cormap(div1,div2):
    """
    calculate the correlation between two Div instances
    Parameters
    ----------
    div1, div2: Div instances dimensioned n * p1 and n * p2 respectively
    These instances must have the same number of rows
    Return
    ------
    cor_div: div instance with matrix of correlation dimensioned p1 * p2
    Example
    -------
    cor_div = cormap(div1,div2)
    """
    
    # Check line compatibility
    ref_n = div1.d.shape[0]
    ref_i = div1.i
    test_n = div2.d.shape[0]
    test_i = div2.i
    if test_n != ref_n:
        raise ValueError('div#' + str(2) + ' does not have the good number of rows. div#' + str(2) + ' has ' + str(test_n) + ' rows but the reference div has ' + str(ref_n) + ' rows')
    if not(np.array_equal(test_i, ref_i)):
        print('Warning: div#' + str(2) + '.i is different from the reference div.i')
    
# meancenter calculation for div 1
    mean_vec1 = np.mean(div1.d, axis=0)
    mmx1 = div1.d - mean_vec1

# meancenter calculation for div 2
    mean_vec2 = np.mean(div2.d, axis=0)
    mmx2 = div2.d - mean_vec2
    
# standardize preprocessing for div1
    xstd1 = np.std(div1.d,axis=0)
    stx1 = mmx1 / xstd1  
    
# standardize preprocessing for div2
    xstd2 = np.std(mmx2,axis=0)
    stx2 = mmx2 / xstd2  

# correlation matrix calculation
    tstx1 = stx1.T
    cor = (1/ref_n)*((tstx1).dot(stx2))
    
    cor_div = copy(div1)
    cor_div.d = cor
    cor_div.i = div1.v
    cor_div.v = div2.v
    cor_div.id = ' matrix correlation'
    
    return cor_div

def covmap(div1,div2):
    """
    calculate the covariance between two Div instances
    Parameters
    ----------
    div1, div2: Div instances dimensioned n * p1 and n * p2 respectively
    These instances must have the same number of rows
    Return
    ------
    cov_div: div instance with matrix of covariance dimensioned p1 * p2
    Example
    -------
    cov_div = covmap(div1,div2)
    """
    
    # Check line compatibility
    ref_n = div1.d.shape[0]
    ref_i = div1.i
    test_n = div2.d.shape[0]
    test_i = div2.i
    if test_n != ref_n:
        raise ValueError('div#' + str(2) + ' does not have the good number of rows. div#' + str(2) + ' has ' + str(test_n) + ' rows but the reference div has ' + str(ref_n) + ' rows')
    if not(np.array_equal(test_i, ref_i)):
        print('Warning: div#' + str(2) + '.i is different from the reference div.i')

    # meancenter calculation for div 1
    mean_vec1 = np.mean(div1.d, axis=0)
    mmx1 = div1.d - mean_vec1

    # meancenter calculation for div 2
    mean_vec2 = np.mean(div2.d, axis=0)
    mmx2 = div2.d - mean_vec2

    # covariance matrix calculation
    tmmx1 = mmx1.T
    cov = (1/ref_n)*((tmmx1).dot(mmx2))

    cov_div = copy(div1)
    cov_div.d = cov
    cov_div.i = div1.v
    cov_div.v = div2.v
    cov_div.id = ' matrix covariance'

    return cov_div
    
def distance(div1,div2):
    """
    calculate the Usual Euclidian distances between two Div instances
    Parameters
    ----------
    div1, div2: Div instances dimensioned n1 * p and n2 * p respectively
    These instances must have the same number of columns
    Return
    ------
    D: matrix n1 x n2 of Euclidian distances between the observations
    Example
    -------
    D = distance(X1,X2);
    """
    
    # Check column compatibility
    ref_n = div1.d.shape[1]
    ref_v = div1.v
    test_n = div2.d.shape[1]
    test_v = div2.v
    if test_n != ref_n:
        raise ValueError('div#' + str(2) + ' does not have the good number of columns. div#' + str(2) + ' has ' + str(test_n) + ' columns but the reference div has ' + str(ref_n) + ' columns')
    if not(np.array_equal(test_v, ref_v)):
        print('Warning: div#' + str(2) + '.v is different from the reference div.v')
    
    nrow1,ncol1 = div1.d.shape[:]
    nrow2,ncol2 = div2.d.shape[:]
    
    aux = np.ones((nrow1,1))
    
    Distance = copy(div1)
    Distance.d = np.zeros((nrow1,nrow2))
    
    for i2 in range(0,nrow2):
        if i2 % 100 == 0:
            print(i2," among ", nrow2)
        delta = (div1.d - aux @ div2.d[i2:i2+1,:]).T
        delta = delta * delta;
        Distance.d[:,i2] = np.sqrt(sum(delta,0))

    Distance.i = div1.i
    Distance.v = div2.i
    Distance.id = 'matrices distance'
    
    return Distance

def group_mean(div,group):
    """
    Compute the means according to a grouping
    Parameters
    ----------
    div: data to be averaged by group
    group: integer giving the group number for each row
    Return
    ------
    Class foo with Div      
        center: div of the averages by group
        group_size: number of observations belonging to the corresponding group

    """
    group_index=np.unique(group.d)
    ngroup=len(group_index)
    #print(ngroup,div.d.shape[1])
    xcenter=np.zeros([ngroup,div.d.shape[1]])
    aux=np.zeros([ngroup,1])
    k=0
    for i in group_index:
        index_group=np.where(group.d == i)[0]
    #         print(index_group.shape)
    #        #print(i,div.d[index_group,:].shape[0])
        aux[k]=np.asarray(div.d[index_group,:].shape)[0]
        xcenter[k,:]=np.mean(div.d[index_group,:])
        k=k+1
    #     print(xcenter)
    center=Div(d=xcenter,i=np.array(group_index),v=div.v) 
    group_size=Div(d=aux,i=np.array(group_index),v='group size')
    Group_mean_type=u.Foo(info='Group_mean',center=center,group_size=group_size) 
    return Group_mean_type
    
def isdiv(obj):
    """
    isdiv tests if obj is a div instance
    Parameters
    ----------
    obj: a python objet
    
    return
    -----
    true if obj has field d, i, v, else otherwise 

    Note
    ------
    The function only verifies that fields d, i, v exist in the object. This
    is sufficient in practice to give confidence that the object is an instance
    of class Div. This function tries to replace the function isinstance of Python
    wich is not correctly working in some situation.
    """
    return(hasattr(obj, 'd') and hasattr(obj, 'i') and hasattr(obj, 'v'))

def randomize(div):
     """
     return div with rows in random order
     Parameters
     ----------
     div: an instance of Div class
     
     return
     ------
     a div matrix with the order of the rows randomized
     
     """
     if(not(isdiv(div))):
         raise ValueError('the entered argument is not an instance of class div')

     rorder=np.random.permutation(np.arange(div.i.shape[0]))
     return(selectrow(div,rorder.tolist()))

def reorder(div1,div2):
    """
    This function makes it possible to realign the rows of div1 and div2, in order
    to have the identifiers corresponding.
    The function discards the observations which are not present in d1v and div2.
    Fails if div1 or div2 contains duplicate (identical) identifiers of rows. 
    The rows in the resulting out Div instances are sorted in the alphabetic order of 
    the rows identifiers (.i)
    
    Parameters
    ----------
    div1, div2 instances of class Div to be reordered

    Returns
    ----------
    out1, out2 : reordered Div instances corresponding to div1 and div2 respectively
    diff1, diff2 : list of rows names with no correspondance found in div1 and div2 repsectively 
    """
    if(not(isdiv(div1))):
         raise ValueError('the entered first argument is not an instance of class div')
    if(not(isdiv(div2))):
         raise ValueError('the entered second argument is not an instance of class div')
    if((np.shape(div1.i)[0])!= (np.shape(np.unique(div1.i))[0])):
        raise ValueError('the first argument has not unique row identifiers. Impossible to reororder')
    if((np.shape(div2.i)[0])!= (np.shape(np.unique(div2.i))[0])):
        raise ValueError('the second argument has not unique row identifiers. Impossible to reororder')
    common=np.intersect1d(div1.i,div2.i)
    list1=list()
    list2=list()
    for index, thisname in enumerate(common):
        row_index1 = np.where(div1.i == thisname)[0]#.ravel().tolist()
        list1.append(row_index1[0])
        row_index2 = np.where(div2.i == thisname)[0]#.ravel().tolist()
        list2.append(row_index2[0])
    out1=selectrow(div1,list1)
    out2=selectrow(div2,list2)

    diff1=list(set(div1.i).difference(set(common))) 
    diff2=list(set(div2.i).difference(set(common))) 
    
    return(out1,out2,diff1,diff2)