# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:42:13 2016

@author: cyyang
"""



from scipy.integrate import quad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### kinematic viscosity
def nu(temperature):
    return 1.79 * 10**-6 /(1 + 0.03368 * temperature + 0.00021 *temperature**2)   ## [m2/s]

### water density
def rho(temperature):
    return -1.8298E-07 * temperature**4 + 4.8685E-08 * temperature**3 - 7.7437E-03 * temperature**2 \
+ 5.0989E-02 * temperature + 9.9988E+02

### dimensionless particle diameter
def d_star(ds, temperature, G=2.65):
    return ds * ((G - 1) * 9.81 / nu(temperature)**2)**(1/3)
    
### fall velocity
def fall_velocity(ds, temperature, G=2.65):
    return (8 * nu(temperature) / ds ) * ((1 + 0.0139 * d_star(ds, temperature, G)**3)**0.5 -1)
  
### shear velocity
def u_star(depth, slope):
    return np.sqrt(9.81 * depth * slope)

### dimensionless shear stress (Shields parameter)
def tau_star(temperature, ds, u):
    return (rho(temperature) * u**2)/(1.65 * 9.81 * 1000 * ds)
    
### Rouse number
def Rouse_number(fall_v, shear_v):
    return fall_v/0.4/shear_v

### 
def E(ds, h):
    return 2 * ds/h

def q_measured(concentration, dischrage, width):
    return (concentration * dischrage / width)/1000

def A(dn, h):
    return dn/h

def J1A(Ro, A):
    def J1(x):
        return ((1-x)/x)**Ro
    return quad(J1,A,1)[0]
def J2A(Ro, A):
    def J2(x):
        return (((1-x)/x)**Ro ) * np.log(x)
    return quad(J2,A,1)[0]

vJ1A = np.vectorize(J1A)
vJ2A = np.vectorize(J2A)
  
def q_bed(q_m, Ro, A, E, h, ds):
    return (q_m * (1 - E)**Ro)/(0.216 * E**(Ro-1) * (np.log(30 * h/ds) * vJ1A(Ro, A) + vJ2A(Ro,A)))


def I1(ro, E):
    def J1(x):
        return ((1 - x)/x)**ro
    return (0.216 * E**(ro-1)/(1 - E)**ro) * quad(J1, E, 1)[0]

def I2(ro, E):
    def J2(x):
        return ((1 - x)/x)**ro * np.log(x)
    return (0.216 * E**(ro-1)/(1 - E)**ro) * quad(J2, E, 1)[0]

    
def q_total(q_b, Ro, E, h, ds):
    vI1 = np.vectorize(I1)
    vI2 = np.vectorize(I2)
    return q_b * (1 + vI1(Ro, E) * np.log(30 * h/ds) + vI2(Ro,E))

def q_unmeasured(qtotal, qmeasured):
    return qtotal - qmeasured





    
def semep_procedure(raw_df, which='sus'):
    dat = raw_df.copy()
    if which == "sus":
        dat['omega'] = fall_velocity(dat['dss'], dat['Temp'])
    elif which == "bed":
        dat['omega'] = fall_velocity(dat['d50'], dat['Temp'])
    dat['ustar'] = u_star(dat["h"], dat["S"])
    dat['ro'] = Rouse_number(dat["omega"], dat["ustar"])
    dat['e'] = E(dat.d50, dat["h"])
    dat['a'] = A(0.1, dat["h"])
    dat['qm'] = q_measured(dat["C"], dat["Q"], dat["W"])   # kg/m-sec
    dat['qb'] = q_bed(dat['qm'], dat['ro'], dat['a'], dat['e'], dat["h"], dat.d65)  # kg/m-sec
    try:
        dat['qt'] = q_total(dat['qb'], dat['ro'], dat['e'], dat["h"], dat.d65)
    except:
        df = dat.loc[~pd.isnull(dat).any(axis=1),:].copy()
        dat.loc[pd.isnull(dat).any(axis=1),'qb'] = 0
        df['qt'] = q_total(df['qb'], df['ro'], df['e'], df["h"], df.d65)
        dat = dat.join(df['qt'])
        dat.loc[pd.isnull(dat).any(axis=1),'qt'] = dat.loc[pd.isnull(dat).any(axis=1),'qm']
    dat['u/w'] = dat["ustar"]/dat["omega"]
    dat['qs'] = dat["qt"] - dat["qb"]
    dat['q'] = dat["Q"]/dat["W"] # unit width discharge in sq-m per sec
    dat['Qm'] = dat["qm"] * dat["W"] * 86400 / 1000    #(tons/day)
    dat['Qt'] = dat["qt"] * dat["W"] * 86400 / 1000    #(tons/day)
    dat['qs/qt'] = dat.qs / dat.qt
    dat['qm/qt'] = dat.qm / dat.qt
    return dat

def raw_extract(path):
    raw_df = pd.read_excel(path, sheetname='sediment measurement', skiprows=4, parse_cols=[5,8,10,11,12,20,26,27,28,29]
                   ,names=["Date", "A", "h", "Q", "Temp", "C", "d50", "d65", "dss", "S"])

    raw_df["Date"] = raw_df["Date"].astype(str)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"].str[:10])
    raw_df[["A", "h", "Q", "Temp", "C", "d50", "d65", "dss", "S"]] = raw_df[["A", "h", "Q", "Temp", "C", "d50", "d65", "dss", "S"]].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    raw_df = raw_df.dropna()
    raw_df['W'] = raw_df["A"]/raw_df["h"]
    raw_df['d50'] = raw_df['d50']/1000
    raw_df['d65'] = raw_df['d65']/1000
    raw_df['dss'] = raw_df['dss']/1000
    raw_df = raw_df[(raw_df.h > 2*raw_df.d50/1000) & (raw_df.h > 0.1)] # remove h < 0.1 m
    return raw_df.reset_index(drop=True)


def export_to_seema(raw_df):
    df = raw_df.copy()
    df['v'] = df['Q']/df['W']
    df['dm'] = 0.1
    df['d10'] = None
    df['frac_wl'] = None
    return df[["C", "Q", "v", 'h', 'dm', 'W', 'S', 'd10', 'd50', 'd65', 'frac_wl', 'dss', 'Temp']]
    


def Einstein(raw_df):
    h, w, q, c, d50, d65, dss, T, S = data_processing(raw_df)
    
    omega = fall_velocity(d50, T)
    ustar = u_star(h, S)
    ro = Rouse_number(omega, ustar)
    e = E(d50, h)
### Einstein method
    def unit_bed_load(tau, fall_v, ds):
        q_star = np.where(tau<0.18, 2.15 * np.e**(-0.391/tau), np.where(tau<0.52, 40*tau**3, 15*tau**1.5))
        return q_star * fall_v * ds
           
    def unit_total_load(q_b, Ro, E, h, ds):
        vI1 = np.vectorize(I1)
        vI2 = np.vectorize(I2)
        return q_b * (1 + vI1(Ro, E) * np.log(30 * h/ds) + vI2(Ro,E))
    
    qb = unit_bed_load(tau_star(T, d50, omega), omega, d50)
    qt = unit_total_load(qb, ro, e, h, d50)
    df = pd.DataFrame()
    df['u*/w'] = ustar/omega
    df['qb'] = qb
    df['qt'] = qt
    df['Q'] = q
    df['Qb'] = qb * 2.65 * w * 86400
    df['Qt'] = qt * 2.65 * w * 86400
    return df


def Engelund_Hansen(raw_df):
    h, w, q, c, d50, d65, dss, T, S = data_processing(raw_df)
    G = 2.65    
    cw = 0.05 * (G/(G - 1)) * q/w/h * S /((G - 1) * 9.81 * d50)**0.5 * (h * S/((G - 1)*d50))**0.5
    df = pd.DataFrame()
    df['cw'] = cw
    df['cmgl'] = G * 10**6 * cw/(G + (1-G)*cw)
    df['qt'] = cw * q
    df['Q'] = q
    df['Qt'] = df.qt * 2.65 * w * 86.400   # cms to tons/day
    return df

def plot(result, save='False', output='output', format='jpg', dpi=200):
    x = result['u*/w']
    y1 = result['qs/qt']
    y2 = result['qm/qt']
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    plt.plot(x, y1, 'o', label=r'$q_s/q_t$')
    plt.plot(x, y2, 'o', label=r'$q_m/q_t$')
    ax.set_xlabel(r'$u_*/\omega$')
    ax.set_ylabel(r'Fraction Total Load')
    ax.set_ylim(0, 1)
    ax.set_xlim(1, 1000)
    ax.set_xscale('log')
    ax.grid(which='both')
    ax.legend(loc='best', numpoints=1)
    if save == True:
        
        plt.savefig(output+'.'+format, format=format, dpi=dpi)
   
def plot_Qvsqmqt(result, save='False', output='output', format='jpg', dpi=200):
    x = result['Q (cms)']
    y = result['qm/qt']
    
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    plt.plot(x, y, 'o')

    ax.set_xlabel(r'Discharge $(m^3/s)$')
    ax.set_ylabel(r'$q_m/q_t$')
    ax.set_ylim(0, 1)

    ax.grid(which='both')
    if save == True:        
        plt.savefig(output+'.'+format, format=format, dpi=dpi)
        
def plot_Qvsqsqt(result, save='False', output='output', format='jpg', dpi=200):
    x = result['Q (cms)']
    y = result['qs/qt']
    
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    plt.plot(x, y, 'o')

    ax.set_xlabel(r'Discharge $(m^3/s)$')
    ax.set_ylabel(r'$q_s/q_t$')
    ax.set_ylim(0, 1)

    ax.grid(which='both')
    if save == True:        
        plt.savefig(output+'.'+format, format=format, dpi=dpi)