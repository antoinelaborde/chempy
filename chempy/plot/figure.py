# -*- coding: utf-8 -*-
"""
28/10/2018
@author: DOMI & ALA
"""

import chempy.utils.util as util
import chempy.utils.classes as classes


import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import OrderedDict


def curve(div, i=None, cmap='hsv', legend_label=None, legend=True): 
    """
    Plots one or several row as curves 
        
    Parameters
    ----------
    div: div or list of div(mandatory)

    i: list (optional, default=None)
        name of the rows to select or indexes of row to select in a list

    Notes
    -----

    if div is a list of div, each div in the list is plot with a different color
    
    """
    mpl.style.use('seaborn')
    # Check if div is a list of div or a div
    if isinstance(div, list):
        for ind, div_ in enumerate(div):
            if isinstance(div_, classes.Div):
                raise ValueError('Element ' + str(ind) + ' of your list is not a Div')
        curve_type = 'list'
    elif not(isinstance(div, classes.Div)):
        raise ValueError('div must be a div instance or a list of div instances')
    else:
        curve_type = 'one'
    
    # Select rows of div
    if i is not None:
        if isinstance(div, list):
            print('i index is ignored when div is a list')
        else:
            div = util.selectrow(div, i)


    # Check if div.v can be interpreted as numericals
    x_vector = util.vfield2num(div)
    if x_vector is None:
        x_vector = np.arange(div.v.shape[0])

    # one div case
    if curve_type == 'one':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_vector, div.d.T)
        plt.show()
    # list of div case
    elif curve_type == 'list':
        cmap_vec = mpl.cm.get_cmap(cmap)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ind = 0
        for div_group, filtername in zip(group_obj.div_list, group_obj.filter_list):
            ind += 1
            ax.plot(x_vector, div_group.d.T,label=filtername, c=cmap_vec(ind/len(group_obj.div_list)))
        # Handle legend
        if legend and legend_label is None:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            leg = plt.legend(by_label.values(), by_label.keys(), title='Legend')
            leg.draggable()
        if legend and legend_label is not None:
            if len(legend_label) != ind:
                raise ValueError('legend_label must contains ' + str(ind) + 'elements')

            leg = plt.legend(legend_label, title='Legend')
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