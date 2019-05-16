import numpy as np
import pandas as pd
import os
from linearmodels import PanelOLS
from linearmodels import PooledOLS
from linearmodels import RandomEffects
from linearmodels import FirstDifferenceOLS
from linearmodels.datasets import jobtraining
from statsmodels.datasets import grunfeld
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

def indfind(state_m1,states):
    for j in range(len(states)):
        identifier= states[j] == state_m1
        try:
            identifier = all(identifier)
        except:
            pass
        if identifier:
            return(j)


def my_meshgrid(y1,y2,z):
    xx,yy,zz = np.meshgrid(y1,y2,z)
    meshed = np.vstack((xx.flatten(),yy.flatten(),zz.flatten()))
    return(meshed.T)


#-------------------------------

def clogit(ydum,x,restx,namesb,print_output):
    cconvb = 1e-6
    myzero = 1e-16
    nobs = int(ydum.shape[0])
    nalt = int(max(ydum) + 1)
    npar = int(x.shape[1]/nalt)
    max_iter = 100
    if npar != len(namesb):
        print("ERROR: Dimensions of x and of names(b0) \nrows(namesb)  do not match ")
        return

    xysum = 0
    for j in range(nalt):
        xbuff = x[:,int(npar*j):int(npar*(j+1))]
        xysum = xysum +  np.dot(np.diag(ydum == j),xbuff)

    iter = 1
    criter = 10
    llike = -nobs
    b0 = np.zeros(int(npar))

    while (criter > cconvb) & (iter < max_iter):
        if (print_output==1):
            print(" \n")
            print("Iteration                = ", iter ,"\n")
            print("Log-Likelihood function  = ", llike ,"\n")
            print("Norm of b(k)-b(k-1)      = ", criter,"\n")
            print(" \n")

        phat = np.zeros((nobs,nalt))
        for j in range(nalt):
            phat[:,j] = np.dot(x[:,int(npar*j):int(npar*(j+1))],b0).flatten() + restx[:,j]

        phat = (phat.T - phat.max(1)).T
        phat = np.exp(phat)
        phat = (phat.T / phat.sum(1)).T

        # Computing xmean
        sumpx = np.zeros((nobs,1))
        xxm = 0
        llike = 0
        for j in range(nalt):
             xbuff = x[:,int(npar*j):int(npar*(j+1))]
             sumpx = sumpx +  np.dot(np.diag(phat[:,j]), xbuff)
             xxm   = xxm + np.dot(xbuff.T, np.dot(np.diag(phat[:,j]), xbuff))
             llike = llike + ( (ydum == j) * np.log( (phat[:,j] > myzero) * phat[:,j]  + (phat[:,j] <= myzero) * myzero  )).sum()

        d1llike = xysum.sum(0) - sumpx.sum(0)
        # d2llike = np.dot(sumpx.T,sumpx)
        # Computing gradient

        # d1llike = xysum - sumpx.sum(0)
        # @ Computing hessian @
        d2llike = - (xxm - np.dot(sumpx.T,sumpx) )

        # @ Gauss iteration @
        invertible = abs(np.linalg.det(d2llike))>0;
        if invertible==1:
            b1 = b0 - np.dot(np.linalg.inv(d2llike),d1llike)
            criter = np.sqrt(np.dot((b1-b0).T,(b1-b0)))
            b0 = b1
            iter = iter + 1
        else:
            b0 = np.zeros((npar,1))
            criter = 0
        if(invertible==1):
            Avarb  = np.linalg.inv(-d2llike)
            sdb    = np.sqrt(np.diag(Avarb))
            tstat  = b0/sdb

            numyj  = np.array( [ 1 * (ydum==j) for j in range(3)])
            logL0  = numyj*np.log(numyj/nobs)
            logL0[logL0==np.nan] = 0
            logL0 = logL0.sum()
            lrindex = 1 - llike/logL0
            if (print_output==1):
                print("---------------------------------------------------------------------")
                print("Number of Iterations     = ", iter)
                print("Number of observations   = ", nobs)
                print("Log-Likelihood function  = ", llike)
                print("Likelihood Ratio Index   = ", lrindex)
                print("---------------------------------------------------------------------")
                print("       Parameter         Estimate        Standard        t-ratios")
                print("                                         Errors" )
                print("---------------------------------------------------------------------")
                for j in range(npar):
                    print(namesb[j],b0[j],sdb[j],tstat[j])
                    print("---------------------------------------------------------------------")
    return([b0,Avarb, invertible])
