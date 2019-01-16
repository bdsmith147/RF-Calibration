# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:05:28 2019

@author: Benjamin
Adapted from: "Another example of fitting a system of ODEs using the lmfit package"
http://people.duke.edu/~ccc14/sta-663-2016/13_Optimization.html
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import complex_ode
from lmfit import minimize, Parameters, report_fit

#filename = 'RabiFlop_-35dBm_20180904'
#data = np.loadtxt(filename + '.csv', dtype=float, delimiter=',', skiprows=2, usecols=(0,1,2))
#data = data[:-2,:]
#time, cloud1, cloud2 = data.T
#cloud0 = 1 - cloud1 - cloud2

def Model(times, ivs, ps):
    '''Using the parameters supplied, computes a complex integration of the
    equations over the time specified durr with step size dt.'''
    try:
        Omeg = ps['Omeg'].value
        delta = ps['delta'].value
        eps = ps['eps'].value
    except:
        Omeg, delta, eps = ps
    def Coefficients(t, c):
        '''c is a vector of c1, c2 and c3; delta is a two-vector of (del21, del32)
        and Omega is another two-vector of (Omeg21, Omeg32)'''
        Ham = -1.0j * np.array([ [ -delta, Omeg/2, 0 ], 
                          [ Omeg/2, eps, Omeg/2], 
                          [ 0, Omeg/2, delta]])
        dcdt = np.dot(Ham, c)
        return dcdt
    durr = times.max()
    dt = times[1] - times[0]
    ivs = np.array([0.0, 0, 1.00], dtype='float')
    odiff = complex_ode(Coefficients)
    odiff.set_initial_value(ivs)
    times, coeffs = [], []
    coeffs.append(ivs)
    while odiff.successful() and odiff.t <= durr:
        sol = odiff.integrate(odiff.t+dt)
        coeffs.append(np.abs((sol * sol)))
    coeffs = np.array(coeffs)
    return coeffs

#################   GENERATE FAKE DATA   #########################
delta = 0
Omeg = 20000
eps = 0
params = [Omeg, delta, eps]

c0 = np.array([0.0, 0, 1.00], dtype='complex')
t1 = 1e-3
dt = 1e-5

times = np.arange(0, t1+dt, dt)
coeffs = Model(times, c0, params)
np.random.seed(54)
noise = np.random.normal(0, 0.05, (len(times), 3))
data = coeffs + noise



#%%
###################   FIT TO FAKE DATA   ###########################

guess = [23000, 1000, 0]
ivs = np.array([0.0, 0.0, 1.0])
t = np.arange(0, t1+dt/10.0, dt/10.0)
initial = Model(t, ivs, guess)
plt.figure()
plt.plot(t, initial, lw=3)
plt.scatter(times, data[:,0], label="0")
plt.scatter(times, data[:,1], label="1")
plt.scatter(times, data[:,2], label="2")
plt.legend()
plt.xlim(0,t1)
plt.title('Initial Guess')
plt.show()
#dt = 1e-5
#t = np.arange(0, t1+dt, dt)

params = Parameters()
params.add('Omeg', value=guess[0], min=0, max=40000)
params.add('delta', value=guess[1])
params.add('eps', value=guess[2], vary=False)
params.add('iv0', value=ivs[0], vary=False)
params.add('iv1', value=ivs[1], vary=False)
params.add('iv2', value=ivs[2], vary=False)

def Residuals(ps, ts, data):
    ivs = np.array([ps['iv0'], ps['iv1'], ps['iv2']])
    model = Model(ts, ivs, ps)
    return np.abs(model - data)


result = minimize(Residuals, params, method='nelder', args=(times, data))
report_fit(result)

t = np.arange(0, t1+dt/10.0, dt/10.0)
final = Model(t, ivs, result.params)
plt.figure()
plt.plot(t, final, lw=3)
plt.scatter(times, data[:,0], label="0")
plt.scatter(times, data[:,1], label="1")
plt.scatter(times, data[:,2], label="2")
plt.legend()
plt.xlim(0,t1)
plt.title('Fitted Parameters')
plt.show()