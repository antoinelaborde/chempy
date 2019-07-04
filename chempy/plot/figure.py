# -*- coding: utf-8 -*-
"""
28/10/2018
@author: DOMI & ALA
"""

import chempy.utils.util as util
import chempy.utils.classes as classes

import numpy as np

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt


from collections import OrderedDict

"""
Idees pour changement de style:
utiliser style pyplot context

"""


def curve(div, row=None, cmap='hsv', legend_label=None, legend=True, ycolor=None): 
    """
    Plots one or several row as curves 
        
    Parameters
    ----------
    div: div or list of div(mandatory)
        div to display
    row: list (optional, default=None)
        name of the rows to select or indexes of row to select in a list
    cmap: str (optional, default='hsv')
        name of the colormap
    legend_label: list of str (optional, None)
        list of str for legend, must be the same number of div in div list
    legend: bool (optional, default=True)
        display legend or not
    ycolor: div (optional, default=None)
        div with values for colors (with one col), only valid for single div input

    Notes
    -----
    if div is a list of div, each div in the list is plot with a different color
    
    """
    mpl.style.use('seaborn')
    # Check if div is a list of div or a div
    
#    print('I am here in chempy\chempy`\plot')
#    print(type(div))
    if isinstance(div, list):
        for ind, div in enumerate(div):
            if not(isinstance(div, classes.Div)):
                raise ValueError('Element ' + str(ind) + ' of your list is not a Div')
        curve_type = 'list'
    elif not(isinstance(div, classes.Div)):
        raise ValueError('div must be a div instance or a list of div instances')
    else:
        curve_type = 'one'
    # Select rows of div
    if row is not None:
        if isinstance(div, list):
            print('row index is ignored when div is a list')
        else:
            div = util.selectrow(div, row)
    # Check and extract ycolor
    if ycolor is not None:
        if not(isinstance(ycolor, classes.Div)):
            raise ValueError('ycolor must be a div')
        if ycolor.d.shape[1] != 1:
            raise ValueError('ycolor must be a div with 1 col (not ' + str(div.d.shape[1]) + ')')
        if curve_type == 'list':
            raise ValueError('ycolor option is not valid for a list of div input')
        if ycolor.d.shape[0] != div.d.shape[0]:
            raise ValueError('ycolor must have the same number of rows than div')
        ycolormap, scalarmap = get_cmap(ycolor.d, cmap)
    else:
        ycolormap = None

    # One div case
    if curve_type == 'one':
        # Check if div.v can be interpreted as numericals
        x_vector = util.vfield2num(div)
        if x_vector is None:
            x_vector = np.arange(div.v.shape[0])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        if ycolormap is None:
            ax.plot(x_vector, div.d.T)
        else:
            for i in range(div.d.shape[0]):
                ax.plot(x_vector, div.d[i,:].T, c=ycolormap[ycolor.d[i,0]])
            fig.colorbar(scalarmap, label = ycolor.v[0])
        plt.show()
        
    # list of div case
    elif curve_type == 'list':
        # Check legend label has to right number of element
        if legend_label is not None:
            if len(legend_label) != len(div):
                print('Warning! legend_label contains ' + str(len(legend_label) + ' elements but should contain ' + str(len(div))))
                legend_label = None
        if legend_label is None:
            legend_label = [str(row) for row in range(len(div))]

        cmap_vec = mpl.cm.get_cmap(cmap)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ind = 0
        for div_group, label in zip(div, legend_label):
            ind += 1
            # Check if div.v can be interpreted as numericals
            x_vector = util.vfield2num(div_group)
            if x_vector is None:
                x_vector = np.arange(div.v.shape[0])

            ax.plot(x_vector, div_group.d.T,label=str(label), c=cmap_vec(ind/len(div)))
        # Handle legend
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            leg = plt.legend(by_label.values(), by_label.keys(), title='Legend')
            leg.draggable()
        plt.show()

def map2(div, col_x, col_y, row=None, cmap='hsv', legend_label=None, legend=True, ycolor=None):
    """
    Plots 2D map
        
    Parameters
    ----------
    div: div or list of div(mandatory)
        div to display
    col_x: int (mandatory)
        index of column of the div as the x axis
    col_y: int (mandatory)
        index of column of the div as the y axis
    row: list (optional, default=None)
        name of the rows to select or indexes of row to select in a list
    cmap: str (optional, default='hsv')
        name of the colormap
    legend_label: list of str (optional, None)
        list of str for legend, must be the same number of div in div list
    legend: bool (optional, default=True)
        display legend or not
    ycolor: div (optional, default=None)
        div with values for colors (with one col), only valid for single div input

    Notes
    -----
    if div is a list of div, each div in the list is plot with a different color
    
    """
    mpl.style.use('seaborn')
    # Check if div is a list of div or a div
    if isinstance(div, list):
        for ind, div_ in enumerate(div):
            if not(isinstance(div_, classes.Div)):
                raise ValueError('Element ' + str(ind) + ' of your list is not a Div')
        map_type = 'list'
    elif not(isinstance(div, classes.Div)):
        raise ValueError('div must be a div instance or a list of div instances')
    else:
        map_type = 'one'
    # Check col_x and col_y
    if not(isinstance(col_x,(int,np.integer))) or not(isinstance(col_y,(int,np.integer))):
        raise ValueError('col_x and col_y must be integers.')
    # Need to check col_x and col_y are in the range of each div to be plot
    # For one div 
    if map_type == 'one':
        if col_x < 0 or col_x >= div.d.shape[1]:
            raise ValueError('col_x (= ' + str(col_x) + ') must be in the range 0 - ' + str(div.d.shape[1]-1))
        if col_y < 0 or col_y >= div.d.shape[1]:
            raise ValueError('col_y (= ' + str(col_y) + ') must be in the range 0 - ' + str(div.d.shape[1]-1))
    elif map_type == 'list':
        for ind, div_ in enumerate(div):
            if col_x < 0 or col_x >= div_.d.shape[1]:
                raise ValueError('col_x (= ' + str(col_x) + ') must be in the range 0 - ' + str(div_.d.shape[1]-1) + 'div#' + str(ind))
            if col_y < 0 or col_y >= div_.d.shape[1]:
                raise ValueError('col_y (= ' + str(col_y) + ') must be in the range 0 - ' + str(div_.d.shape[1]-1) + 'div#' + str(ind))
    # Select rows of div
    if row is not None:
        if isinstance(div, list):
            print('row index is ignored when div is a list')
        else:
            div = util.selectrow(div, row)
    # Check and extract ycolor
    if ycolor is not None:
        if not(isinstance(ycolor, classes.Div)):
            raise ValueError('ycolor must be a div')
        if ycolor.d.shape[1] != 1:
            raise ValueError('ycolor must be a div with 1 col (not ' + str(div.d.shape[1]) + ')')
        if map_type == 'list':
            raise ValueError('ycolor option is not valid for a list of div input')
        if ycolor.d.shape[0] != div.d.shape[0]:
            raise ValueError('ycolor must have the same number of rows than div')
        ycolormap, scalarmap = get_cmap(ycolor.d, cmap)
    else:
        ycolormap = None

    # One div case
    if map_type == 'one':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if ycolormap is None:
            ax.scatter(div.d[:,col_x], div.d[:,col_y])
        else:
            for i in range(div.d.shape[0]):
                ax.scatter(div.d[i,col_x], div.d[i,col_y], c=ycolormap[ycolor.d[i,0]])
            fig.colorbar(scalarmap, label = ycolor.v[0])
        ax.set_xlabel(str(div.v[col_x]))
        ax.set_ylabel(str(div.v[col_y]))
        plt.show()
    
    # list of div case
    elif map_type == 'list':
        # Check legend label has to right number of element
        if legend_label is not None:
            if len(legend_label) != len(div):
                print('Warning! legend_label contains ' + str(len(legend_label) + ' elements but should contain ' + str(len(div))))
                legend_label = None
        if legend_label is None:
            legend_label = [str(row) for row in range(len(div))]

        cmap_vec = mpl.cm.get_cmap(cmap)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ind = 0
        for div_group, label in zip(div, legend_label):
            ind += 1
            ax.scatter(div_group.d[:,col_x], div_group.d[:,col_y],label=str(label), c=cmap_vec(ind/len(div)))
        # Handle legend
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            leg = plt.legend(by_label.values(), by_label.keys(), title='Legend')
            leg.draggable()
        plt.show()


def map(div,col1,col2,margin=0.1,group='',fontsize=20):
   """
   Plots two columns as scatter plot using the identifiers of rows as label
   Parameters
   ----------
   div: div file  
   col1, col2: indices of columns to be plotted
   margin: proportion of non used part of the axis . This is useful for
   the correct plotting of labels with long string names
   """
#   N =5
#   params = pl.gcf()
#   plSize = params.get_size_inches()
#   params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
   xcolor=([0, 0, 0],[1, 0, 0],[0, 0, 1],[0, 0.7, 0],\
   [0.5, 0.5, 0],[0.5, 0, 0.5],[0, 0.5, 0.5],[0.25, 0.25, 0.25],\
   [0.5, 0, 0],[0, 0.5, 0],[0, 0.5, 0],[0.1, 0.2, 0.3],\
   [0.3, 0.2,0.1],[0.5,0.5,0.8],[0.1,0.8,0.1])
   ncolor=len(xcolor)  
   xmin=np.min(div.d[:,col1])
   xmax=np.max(div.d[:,col1])
   deltax=(xmax-xmin)*margin    
   ymin=np.min(div.d[:,col2])
   ymax=np.max(div.d[:,col2])
   deltay=(ymax-ymin)*margin
   plt.axis([xmin-deltax,xmax+deltax,ymin-deltay,ymax+deltay])
   for i in range(0,div.d.shape[0]):
        if(group==''):        
            #print('iam here')            
            plt.text(div.d[i,col1],div.d[i,col2],div.i[i])
        else:
            thiscolor=xcolor[(group.d[i]+1)%ncolor]    
            plt.text(div.d[i,col1],div.d[i,col2],div.i[i],color=thiscolor,fontsize=20)
   plt.xlabel(div.v[col1])   
   plt.ylabel(div.v[col2])
   plt.show()

def get_cmap(y, cmap):
    """
        get_cmap sends back color information according to y values

        Parameters
        ----------
        y: array (mandatory)
            array of values to use for color
        cmap: str (mandatory)
            cmap name
        Returns
        -------
        colors: array
    """

    normcolor = mpl.colors.Normalize(vmin = np.nanmin(y), vmax = np.nanmax(y))
    scalarmap = mpl.cm.ScalarMappable(norm = normcolor, cmap = cmap)
    unique_y = np.asarray(np.unique(y[~np.isnan(y)]))
    colors = {}
    for yval in unique_y:
        colors[yval] = scalarmap.to_rgba(yval)
    scalarmap.set_array([])
    return colors, scalarmap