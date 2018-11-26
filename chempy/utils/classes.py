# -*- coding: utf-8 -*-
"""
2018/20/28
@author: ALA
"""
# Import ######################################
import numpy as np
from copy import deepcopy

"""
class Div():
class Foo():
"""

class Div():
    """ Creation and elementary manipulatiosn of files
    in the d, i, v format
    ---------------------        
    
    """
    
    def __init__(self, d = [], i = [], v = [], id = ''):
        """ Init function for Div
            ---------------------        
            d, i and v are set to empty array by default
        """
        # .d field is considered as the reference
        self.d = np.array(d)

        # Add a dimension to d if needed
        if len(self.d.shape) != 2:
            self.d = np.expand_dims(self.d, 1)
            
        # Manage i field
        if isinstance(i, str):
            i_field = [i]
        else:
            i_field = list(i)
        if len(i_field) == 0:
            print('i field is empty, numerical row names have been added to div.')
            i_val = list(range(self.d.shape[0]))
        elif len(i_field) != self.d.shape[0]:
            print('Warning (' + id + ')! the i field you provide has incorrect number of elements (i has ' + str(len(i_field)) + ' and should have ' + str(self.d.shape[0]) + '). Numerical row names have been added to div. ')
            i_val = list(range(self.d.shape[0]))
        else:
            i_val = i_field
        self.i = np.array(i_val).astype('str')

        # Manage v field
        if isinstance(v, str):
            v_field = [v]
        else:
            v_field = list(v)
        if len(v_field) == 0:
            print('v field is empty, numerical col names have been added to div.')
            v_val = list(range(self.d.shape[1]))
        elif len(v_field) != self.d.shape[1]:
            print('Warning (' + str(id) + ')! the v field you provide has incorrect number of elements (v has ' + str(len(v_field)) + ' and should have ' + str(self.d.shape[1]) + '). Numerical col names have been added to div. ')
            v_val = list(range(self.d.shape[1]))
        else:
            v_val = v_field
        self.v = np.array(v_val).astype('str')
            
        self.id = id
        # Champ preprocessing (.p pas assez signifiant ?)
        # self.p=[]

class Foo:
    """ Dummy variable for basic structure
    ---------------------        
    Comment:
    ALA: check_div to be removed if not necessary
    
    """
    def __init__(self, check_div=False, **kwargs):
        """ Init function for Foo
            ---------------------        
            check_div: bool (optional, default=False)
                if True, check that all fields are Div structure
        """
        if check_div:
            # Check arg are div
            for key in kwargs.keys():
                if type(kwargs[key]).__name__ != 'Div':
                    print(key + ' arg type is ' + type(kwargs[key]).__name__ + ' but Foo expects Div class')
        self.__dict__.update(kwargs)

    def field(self):
        """
        get_field returns all the fields of Foo in list
        Parameters
        ----------

        Returns
        -------
        field_list: list of str
        """
        return list(self.__dict__.keys())
    
    def copy(self):
        """
        copy returns a deepcopy of the instance
        Parameters
        ----------

        Returns
        -------
        Foo
        """
        return deepcopy(self)