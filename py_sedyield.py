# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:50:57 2019

@author: cyyang
"""

import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import fsolve
import scipy.stats as sp


def trainLinear(x,T):
    """Linear regression model"""
    T = T.values.reshape((-1,1))
    X = sm.add_constant(np.log(x))
    model = sm.OLS(np.log(T),X).fit()
    return model

def use(x,model):
    X = sm.add_constant(np.log(x))
    return np.dot( X, model.params )


def get_regression_params(model):
    a = np.e**model.params[0]
    b = model.params[1]
    return a, b

def rmse(P,T):
    return np.sqrt(np.mean( (P-T)**2 ))

def mape(P, T):
    return (np.abs(P - T)/T).mean()

def ccc(P, T):
    n = T.shape[0]
    P_mean = P.mean()
    T_mean = T.mean()
    Sxy = np.sum((P - P_mean) * (T - T_mean)) / (n - 1)
    Sx_sq = np.sum((P - P_mean)**2) / (n - 1)
    Sy_sq = np.sum((T - T_mean)**2) / (n - 1)
      
    CCC = 2 * Sxy / (Sx_sq + Sy_sq + (P_mean - T_mean)**2)
    return CCC

def r_squared(P, T):
    P_mean = P.mean()
    T_mean = T.mean()  
    return np.sum((P - P_mean) * (T - T_mean)) / np.sqrt(np.sum((P - P_mean)**2) * np.sum((T - T_mean)**2))
# for plotting
def myLogFormat(y,pos):
    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)




def flowDuration(dailyQ):
    dailyQ = dailyQ.dropna(subset=['Q'])
    dailyQ = dailyQ[dailyQ['Q'] >= 0]
    dailyQ = dailyQ.sort_values('Q',ascending=False)
    dailyQ['rank'] =dailyQ['Q'].rank(ascending=False,method='min')
    dailyQ['P'] = dailyQ['rank']/(dailyQ.date.count()+1) * 100.0
    dailyQ = dailyQ.drop(dailyQ.columns[[0,2]], axis=1)
    return dailyQ.reset_index(drop=True)


def flowDurationBins(flowDurationCurve):
    """Input: flow Duration Curve"""
    df = flowDurationCurve
    
    def extrapolate(dataframe):
        # find indexes of unknown 
        nanIndex = dataframe.Q[np.isnan(dataframe.Q)==True].index.tolist()
        try:
            if len(nanIndex)>0:
                for i in sorted(nanIndex,reverse=True):
                    indexlist = dataframe.index.tolist()
                    j = indexlist.index(i)
                    q1 = dataframe.iat[j+1,0]
                    q2 = dataframe.iat[j+2,0]
                    p1 = dataframe.iat[j+1,1]
                    p2 = dataframe.iat[j+2,1]
                    p0 = dataframe.iat[j,1]
                    q0 = q2 - (p2-p0)*(q2-q1)/(p2-p1)
                    dataframe.loc[i,'Q'] = q0
            return dataframe
        except:
            return dataframe    

    
    mid = [0.01,0.06,0.3,1,3.25,10,20,30,40,50,60,70,80,90,97.5]
    d = [0.02,0.08,0.4,1,3.5,10,10,10,10,10,10,10,10,10,5]
    a = np.empty(len(mid))
    a[:] = np.nan
    d_ = {'Q':a,'P':mid}
    df2 = pd.DataFrame(data=d_)
       
    result = pd.concat([df,df2]).drop_duplicates('P')
    result = result.sort_values('P')
    result2 = result.set_index('P')
    result2 = result2.interpolate(method='index')
    result2.loc[:,'P'] = result2.index
       
    result2 = result2.reset_index(drop=True)

    extrapolate(result2)
    
    selected = result2.loc[result2['P'].isin(mid)].reset_index(drop=True)
    selected.loc[:,'D'] = d
    return selected





def FDSRC(fdc, src):
    df = flowDurationBins(fdc)
    a = np.e**src.params[0]
    b = src.params[1]
    df['Qt'] = a * df['Q']**b
    df['QtXD'] = df['Qt'] * df["D"]/100
    return df




def graphical_method(fdc, f = 1.5, plot=False):
    fdc = fdc.dropna()
    Q = fdc['Q']
    P = fdc['P']
    
    x = Q[Q > f * Q.mean()]
    y = -np.log(P[Q > f*Q.mean()]/100)
    

    x = np.log(x)
    x = sm.add_constant(x)
    ols = sm.OLS(np.log(y), x).fit()

    (a, b) = (np.e**ols.params[0], ols.params[1])

    ahat = (1 / a)**(1/b)
    bhat = 1 / b
    
    if plot:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(np.log(Q), np.log(-np.log(P/100)))
        ax.set_xlabel('$\ln Q$ (ft$^3$/s)')
        ax.set_ylabel('$\Pi = \ln(-\ln E)$')
        ax.grid()

        ax.plot(x, np.log(a)+x*b, 'k--')
        #ax.set_ylim(ymin=-0.5)
        ax.annotate('$\Pi=${:4.2f} + {:4.2} $\ln Q$'.format(a, b), (1.6, 1.7), fontsize='large')
    return ahat, bhat

def method_of_moments(Q):

    mean = Q.mean()
    sm   = (Q**2).mean()
    #print('Qmean = {:.{prec}f}'.format(mean, prec=2) + ', Qvar = {:.{prec}f}'.format(sm, prec=2))
    def b_func(b):
        return (sm/mean**2) - gamma(1 + 2/b)/gamma(1 + 1/b)**2
    b = fsolve(b_func, 0.1)
    a = (gamma(1 + 1/b)/mean)**b

    ahat = (1 / a)**(1/b)
    bhat = 1 / b
    return ahat[0], bhat[0]


def meanAnnualSedDischarge(A, B, ahat, bhat):
    """A: sediment rating curve coefficient
       B: sediment rating curve exponent
       ahat: fdr transform coefficient
       bhat: fdr transform exponent
    """
    return A*ahat**B*gamma(1 + B*bhat) * 365.25

def cdfDiff(dailyQ, src, ahat, bhat):
    
    b = 1 / bhat    
    a = ahat**(-b)
 
    dailyQ = dailyQ.dropna()
    
    abar = np.e**src.params[0]
    bbar = src.params[1]    
    sort = dailyQ.sort_values(by='Q')
    sort = sort[sort['Q'] >= 0]
    Q = sort['Q']
    Qcf = sort['Q'].cumsum() / sort['Q'].sum()
    Qmean = sort['Q'].mean()
    normQ = Q/Qmean
    
    Qs = abar * Q**bbar
    Qscf = Qs.cumsum() / Qs.sum()
    Qsmean = Qs.mean()
    normQs = Qs/Qsmean
    
    i = a * Q **b
    ahat = (1/a) **(1/b)
    bhat = (1/b)
    P1 = sp.gamma.cdf(i, bhat + 1)
    P2 = sp.gamma.cdf(i, bhat * bbar + 1)
    
    DQ = np.abs(Qcf - P1).max()
    DQs = np.abs(Qscf - P2).max()
    edmQ = np.sum(np.multiply(np.abs((Qcf - P1).values[1:]), np.diff(normQ/normQ.max())))
    edmQs = np.sum(np.multiply(np.abs((Qscf - P2).values[1:]), np.diff(normQs/normQs.max())))
    return DQ, DQs, edmQ, edmQs