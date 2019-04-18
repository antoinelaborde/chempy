# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:55:10 2019

@author: DOMI
"""

#>>> list(range(3, 6))            # normal call with separate arguments
#[3, 4, 5]
#>>> args = [3, 6]
#>>> list(range(*args))            # call with arguments unpacked from a list
#[3, 4, 5]
from scipy import stats as st

#import util as u
#import import_ as ip
#import figure as fig
import numpy as np

def kruswal (X,g):
     """
     Computes a kruskal-wallis test on all the columns of a Div instance.
     The groups are defined in a Div vector of integers g. The observations (rows in X)
     having the same number in g belongs to the same group.
     The Kruskal-Wallis H-test tests the null hypothesis that the population
     median of all of the groups are equal.  It is a non-parametric version of
     ANOVA.
     
     
     Input arguments
     ---------------
     X: Div instance with n rows and p columns
     g: Div instance with nrows (integer) and 1 column

     Output argument
     ---------------
     Div instance with 2 rows and p columns
         row 0: float
             kruskal-wallis H statistic
         row 1: float
             p value
     
     References
     ----------
     This function is basically an entrance to the function scipy.kruskal for 
     the case in which we have many kruskal-wallis tests to be computed on theis
     columns on a matrix X, with the same groups. 
     This is useful when processng series of signals such as spectra. 
     
     Note
     ----
     When X contains many rows, it may be useful to compare the obtained results 
     with the ones observed after randomization of g .
     
     Examples
     --------
     X=import.read2div("X1.csv") #Div structure of signals
     g=grouping(X,[0,1])# groups builds from two first characters in row identifiers
     kres=kruswal(X,g.div_group_index)
     fig.curve(kres,[0]) #curve of H statistics values
     fig.curve(kres,[1]) #curve of pvalues
    
     
     """
     
     xdata=X.d
     n,p=np.shape(xdata)
     l1=len(g.v)
     l2=len(g.d);
     mykeys=np.unique(g.d)
     n1=np.shape(mykeys)
     if l1!=1:
         raise ValueError("'second param' must have only one columns")
     if(l2!=n):
         raise ValueError("'the first and second argument must have the same number of rows")
     res=np.zeros((2, p),dtype=np.float64)
     for i in range(0,p):
         thisrow=xdata[:,i]
         xarg=[]
         ##print(i)
         for j in range(0, n1[0]):
             extracted=np.extract(g.d==mykeys[j],thisrow)
             #print(i,j,np.shape(extracted))
             xarg.append(extracted)
         res[0,i],res[1,i]=st.kruskal(*xarg)
     varname=np.array(['H stat','Pvalue']) 
     return(cp.Div(res,varname,X.v))
 
def anavar1(X,g):
     xdata=X.d
     n,p=np.shape(xdata)
     l1=len(g.v)
     l2=len(g.d);
     mykeys=np.unique(g.d)
     n1=np.shape(mykeys)
     if l1!=1:
         raise ValueError("'second param' must have only one columns")
     if(l2!=n):
         raise ValueError("'the first and second argument must have the same number of rows")
     res=np.zeros((2, p),dtype=np.float64)
     for i in range(0,p):
         thisrow=xdata[:,i]
         xarg=[]
         ##print(i)
         for j in range(0, n1[0]):
             extracted=np.extract(g.d==mykeys[j],thisrow)
             #print(i,j,np.shape(extracted))
             xarg.append(extracted)
         res[0,i],res[1,i]=st.f_oneway(*xarg)
     varname=np.array(['F stat','Pvalue']) 
     return(cp.Div(res,varname,X.v))
    
#scipy.stats.f_oneway
        

if __name__=='__main__':
    import chempy as cp
    import util as u
    import import_ as ip
    import figure as fig
    X=ip.read2div("X1.csv") 
    g=u.grouping(X,[0,1])
    kres=kruswal(X,g.div_group_index)
    fig.curve(kres,[0])
    fig.curve(kres,[1])
    ano=anavar1(X,g.div_group_index)
    fig.curve(ano,[0])
    fig.curve(ano,[1])