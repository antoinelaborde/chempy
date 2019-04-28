# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:55:10 2019

@author: DOMI
"""

from scipy import stats as st
import numpy as np

import chempy.utils.util as util


def kruswal(X,g):
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
    
    cor_div = util.copy(div1)
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
    
    cov_div = util.copy(div1)
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
    
    Distance = util.copy(div1)
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