# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:55:10 2019
examined on 04 07 2019 ==============================
@author: DOMI
"""

from scipy import stats as st

import util as u
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
     """
     Computes an analysis of variance with one factor on all the columns of a Div instance.
     The groups are defined in a Div vector of integers g. The observations (rows in X)
     having the same number in g belongs to the same group.
     The anova test tests the null hypothesis from the Fisher F 
      
     Input arguments
     ---------------
     X: Div instance with n rows and p columns
     g: Div instance with nrows (integer) and 1 column

     Output argument
     ---------------
     Div instance with 2 rows and p columns
         row 0: float
             Anova F statistic
         row 1: float
             p value
     
     References
     ----------
     This function is basically an entrance to the function scipy.f_oneway for 
     the case in which we have many anova tests to be computed on theis
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
     ano=anavar1(X,g.div_group_index)
     fig.curve(ano,[0])
     fig.curve(ano,[1])
     """

     xdata=X.d
     n,p=np.shape(xdata)
     l1=len(g.v)
     l2=len(g.d);
     mykeys=np.unique(g.d)
     n1=np.shape(mykeys)
     if l1!=1:
         raise ValueError("'second parameter' must have only one columns")
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
    
def anavar2(X,g1,g2,interaction=True):
     """
     analysis of variance with two factors with repeats on all the columns of a Div instance.
     The groups are defined in two Div vectors of integers g1,g2.
     The observations (rows in X) having the same number in g belongs to the same group.
     The anova test tests the null hypothesis from the Fisher F 
      
     Input arguments
     ---------------
     X: Div instance with n rows and p columns
     g1: Div instance with n rows (integer) and 1 column
     g2: Div instance with n rows (integer) and 1 column
     interaction: True if interaction is asked, False otherwise

     Output argument
     ---------------
     Foo class with fields
         Div instance F with 3 rows and p columns
             row 0: float
                 Anova F statistic for effect 1
             row 1: float 
                 Anova F statistic for effect 2
             if(interaction asked) row 2: float
                 Anova F statistic for interaction 1 X 2
         Div instance P with 3 rows and p columns
             row 0: float
                 P value (Prob>F)  for effect 1
             row 1: float 
                 P value (Prob>F) for effect 2
             if(interaction asked) row 2: float
                 P value (Prob>F) for interaction 1x2
         list dl
             degree of freedom
     Notes
     ----
     1) When X contains many rows, it may be useful to compare the obtained results 
     with the ones observed after randomization of g1 and g2 .
     2) if the number of repeats is differents between ixj cells, the results
     are only approximate.
     """

    
    #     xdata=X.d
     n,m=np.shape(X.d)
     l1=len(g1.v)
     l2=len(g2.v)
     if l1!=1:
         raise ValueError("'second param' must have only one variable")
     if l2!=1:
         raise ValueError("'third param' must have only one variable")
     if(n!=len(g1.d)):
         raise ValueError("'the first and second argument must have the same number of rows")
     if(n!=len(g2.d)):
         raise ValueError("'the first and third argument must have the same number of rows")
     if(interaction):
         Fx=np.zeros((3, m),dtype=np.float64)
         Px=np.zeros((3, m),dtype=np.float64)
     else:
         Fx=np.zeros((2, m),dtype=np.float64)
         Px=np.zeros((2, m),dtype=np.float64)    
     for col in range(0,m):
         thiscol=X.d[:,[col]]
    #     print(np.shape(X.d[:,0]))
    #     print(np.shape(thiscol))
         Xbarxxx=np.mean(thiscol)
         ng1=np.amax(g1.d)+1 #group numbers start at 0 !!
         ng2=np.amax(g2.d)+1
    #     print(Xbarxxx,ng1,ng2)
         Xbarijx=np.empty((ng1,ng2))
         nij=np.empty((ng1,ng2))
         for i in range(0,ng1):
           for j in range(0,ng2):
    #         print(i,j)
             thiscell=thiscol[(g1.d==i)*(g2.d==j)] #brr ! "and" only working with * (multiplication)
             #print(thiscell)
             Xbarijx[i,j]=np.mean(thiscell)
             nij[i,j]=len(thiscell)
         p,q=np.shape(Xbarijx)
         Xbarixx=np.empty(p)
         Xbarxjx=np.empty(q)
         for i in range(0,p):
             Xbarixx[i]=sum(Xbarijx[i,:])/q;
         for j in range(0,q):
            Xbarxjx[j]=sum(Xbarijx[:,j])/p;
    #     print(Xbarijx)
    #     print(Xbarixx)
    #     print(Xbarxjx)
         SSa=0;
         ni=np.sum(nij,axis=1)
#         print(ni)
         for i in range(0,p):
    #        ni=np.sum(nij,axis=1)
            SSa=SSa+(Xbarixx[i]-Xbarxxx)*(Xbarixx[i]-Xbarxxx)*ni[i] #sum of square a
        # SSa=SSa*p
        # print(SSa)
         SSb=0;
         nj=np.sum(nij,axis=0)
        # print(nj)
         for j in range(0,q):
            SSb=SSb+(Xbarxjx[j]-Xbarxxx)*(Xbarxjx[j]-Xbarxxx)*nj[j] #sum of square b
        # print(SSb)     
         SSab=0;
         if(interaction):
             for i in range(0,p):
                 for j in range(0,q):
                     thissum=Xbarijx[i,j]-Xbarixx[i]-Xbarxjx[j]+Xbarxxx; #sum of square interaction
                     SSab=SSab+nij[i,j]*thissum*thissum;
         
         #print(SSab)
        
         aux=thiscol-Xbarxxx
         total=np.dot(aux[:,0],aux[:,0])
         SSr=total-SSa-SSb-SSab
         #print(SSa,SSb,SSab,SSr)
         dfa=p-1
         dfb=q-1
         MSa=SSa/dfa
         MSb=SSb/dfb
         if(interaction):
             dfab=dfa*dfb
             dfr=n-dfab-dfa-dfb-1
             MSab=SSab/dfab
             MSr=SSr/dfr
             Fab=MSab/MSr
         else:
             dfr=n-dfa-dfb-1
             MSr=SSr/dfr
             MSab=0 #no interaction
             Fab=0
             dfab=0
         Fa=MSa/MSr
         Fb=MSb/MSr
         #print(MSa,MSb,MSab,MSr)
         #print(Fa,Fb,Fab)
#        # print('Fa =',Fa,'Fb=',Fb)
#        # print('dfa=','fb=',dfb)
         Pa=1-st.f.cdf(Fa,dfa,dfr)
         Pb=1-st.f.cdf(Fb,dfb,dfr)
         Pab=1-st.f.cdf(Fab,dfab,dfr)
         #print(round(Pa,4),round(Pb,4),round(Pab,4))

         Fx[0,col]=Fa
         Fx[1,col]=Fb
         Px[0,col]=Pa
         Px[1,col]=Pb
         if(interaction):
             Fx[2,col]=Fab
             Px[2,col]=Pab
         dl=[]
     dl.append([str(dfa)+'/'+str(dfr)])
     dl.append([str(dfb)+'/'+str(dfr)])
     if(interaction):
         dl.append([str(dfab)+'/'+str(dfr)])
         varname=np.array(['X1','X2','X1*X2'])
         info='ANOVA 2 factors, with interaction'
     else:
         varname=np.array(['X1','X2'])
         info='Anova 2 factors, no interaction'
     F=cp.Div(Fx,varname,X.v)
     P=cp.Div(Px,varname,X.v) 
     ANOVA= util.Foo(info=info,F=F,P=P,dl=dl) 
     return(ANOVA)
     

     
         
         
         
     
     
#%SCEab=SCEab*(p-1)*(q-1);
#
#SCEb
#SCEab
#
#        

#if __name__=='__main__':
#    import chempy as cp
#    import util as u
#    import import_ as ip
#    import figure as fig
#    X=ip.read2div("X1.csv") 
#    g=u.grouping(X,[0,1])
#    kres=kruswal(X,g.div_group_index)
#    fig.curve(kres,[0])
#    fig.curve(kres,[1])
#    ano=anavar1(X,g.div_group_index)
#    fig.curve(ano,[0])
#    fig.curve(ano,[1])