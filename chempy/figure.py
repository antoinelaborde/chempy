# -*- coding: utf-8 -*-
"""
28/10/2018
@author: DOMI & ALA
"""

import chempy.utils.util as util
#import chempy.utils.classes as classes
import numpy as np
from classes import Div, Foo
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import figure as fig
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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
    print('figure curve from chempy\chempy\plot')
    mpl.style.use('seaborn')
    # Check if div is a list of div or a div
    if isinstance(div, list):
        for ind, div_ in enumerate(div):
            if not(isinstance(div_, classes.Div)):
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

def show_vector(div,row,xfontsize=10):
   """
   Represents a row of a matrix as a succession identifiers
   Parameters
   -----------
   div : div matrix
   row(integer) : the row to be represented
   xfontsize(optional): size of the font (default 10)
...The identifiers of the columns are plotted with X being the index of the variable and Y the actual
...value of the variable for the selected row "nrow"
...
...Main  use : examining the output of "anavar1" and "anovan1" functions on
   discrete variables
   """
#   N =5
#   params = pl.gcf()
#   plSize = params.get_size_inches()
#   params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
   margin=0.1
   if(not(isdiv(div))):
         raise ValueError('the entered first argument is not an instance of class div')
   thisrow=selectrow(div,[row]);
   lenx=thisrow.d.size;
   #print(thisrow.d[0,5])
   xmin=0
   xmax=thisrow.d.size
   deltax=(xmax-xmin)*margin    
   ymin=np.min(thisrow.d)
   ymax=np.max(thisrow.d)
   deltay=(ymax-ymin)*margin
   plt.axis([xmin-deltax,xmax+deltax,ymin-deltay,ymax+deltay])
   for i in range(0,lenx):
      plt.text(i,thisrow.d[0,i],thisrow.v[i],fontsize=xfontsize)
   
   plt.title(thisrow.i[0])   
   plt.xlabel('Rank of the variables')
   plt.ylabel('Intensity')
   plt.show()
   
def dendro(div,cut=30):
    """
    dendro					- dendrogram using euclidian metric and Ward linkage
    parameters
    ----------
    div : a div matrix
    cut : level of cutting the dendrogram (default :30)
    
    returns
    -------
    dendro_obj with fields
        info_obj  :'dendrogram'
        group     : identifier of the group of each observtion
        center    :barycenter of each group
        group size: number of observations in each group 
    The function displays a dendrogram possibly cut at a given level
    and gives the values od the obtained clustersmy
    """
    X1=div.d
    Z=linkage(X1, 'ward')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    R=dendrogram(Z,leaf_rotation=90.,leaf_font_size=12.,labels=div.i,truncate_mode='lastp',p=cut,
               show_leaf_counts=False,show_contracted=False)
    plt.show()
    T = fcluster(Z, cut, 'maxclust')
    #print(T.size)
#    clustername=str(range(1,(cut-1))
    Tx=Div(T,div.i,'group')
    center,group_size=util.group_mean(div,Tx)
    info_obj = util.Foo('dendrogram') 
    dendro_obj = util.Foo(info_obj=info_obj, group=Tx, center=center, group_size=group_size)

    return dendro_obj