# -*- coding: utf-8 -*-
"""
28/10/2018
@author: DOMI & ALA
"""

import matplotlib.pyplot as plt


def curve(div,indices): 
    """
    Plots one or several row as curves 
    Tests if the variables names can be interpreted as values for the x axis)
        
    Parameters
    ----------
    div: div file  
    indices: indices of the row to be plotted (list of integer or a single integer)   
    """
    
    try :
        floats=[float(x) for x in div.v]
    except ValueError:
        floats=range(0,len(div.v))
    if isinstance(indices,int):
        plt.plot(floats,div.d[indices,:])        
            #pl.title(div.i[indices])   
    else:
        for row in indices:
                plt.plot(floats,div.d[row,:])
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