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
from lmfit.models import LorentzianModel, VoigtModel
filename = 'trialData'
data = np.loadtxt(filename + '.csv', dtype=float, delimiter=',', skiprows=2, usecols=(0,1,2))
data = data[:-2,:]
bias, cloud1, cloud2 = data.T
cloud0 = 1 - cloud1 - cloud2
data = np.stack((cloud0, cloud1, cloud2), axis=1)

calibration = 0.47e3 #kHz/A

mod = LorentzianModel()
#mod = VoigtModel()
pars = mod.guess(cloud1, x=bias)
out = mod.fit(cloud1, pars, x=bias)
#print(out.fit_report(min_correl=0.25))

bias = (bias - out.params['center'].value) * calibration
out.params['center'].value = 0
out.params['center'].stderr = out.params['center'].stderr * calibration
out.params['sigma'].value = out.params['sigma'].value * calibration
out.params['sigma'].stderr = out.params['sigma'].stderr * calibration
out.params['amplitude'].value = out.params['amplitude'].value * calibration
out.params['amplitude'].stderr = out.params['amplitude'].stderr * calibration
#print(out.fit_report(min_correl=0.25))
b = np.linspace(bias[0], bias[-1], 1000)
new = mod.eval(out.params, x=b)
print('Center: %.3f +/- %.3f'%(out.params['center'].value, out.params['center'].stderr))

plt.figure()
plt.plot(b, new, lw=2)
plt.scatter(bias, cloud1)
plt.scatter(bias, cloud0)
plt.scatter(bias, cloud2)
plt.xlabel('Detuning (kHz)')
plt.ylabel('Population')
plt.title('Initial Fit')
plt.show()

#%%

def Model(delta_arr, ps):
    '''Using the parameters supplied, computes a complex integration of the
    equations over the time specified durr with step size dt.'''
    try:
        Omeg = ps['Omeg'].value
        eps = ps['eps'].value
    except:
        Omeg, eps = ps
    #evalues = []
    evectors = []
    for delta in delta_arr:    
        Ham = np.array([ [ -delta, Omeg/2, 0 ], 
                          [ Omeg/2, eps, Omeg/2], 
                          [ 0, Omeg/2, delta]])
        evals, evecs = np.linalg.eig(Ham)
        state = 0
        evectors.append(np.abs(evecs[state]*evecs[state]))
    coeffs = np.array(evectors)
    coeffs = Sort(coeffs, delta_arr)
    return coeffs

def Sort(tab, delta_arr):
    length = tab.shape[0]
    width = tab.shape[1]
    Del = delta_arr[1] - delta_arr[0]
    for col in range(width):
        for row in range(2, length):
#            dist = np.abs(tab[row,:] - tab[row-1, col])
            slope = (tab[row-1, col] - tab[row-2, col])/Del
            guess = slope*Del + tab[row-1, col]
            dist = np.abs(tab[row,:] - guess)
            elem = np.argmin(dist)
            if elem != col:
                tab[row, col], tab[row, elem] = tab[row, elem], tab[row, col]
    return tab
            


def Residuals(ps, d_arr, data):
    model = Model(d_arr, ps)
    return (model - data)**2

#################   GENERATE FAKE DATA   #########################
#dd = 1000
#dmax = 50000
#delta_arr = np.arange(-dmax, dmax + dd, dd)
#Omeg = 40000
#eps = -40000
#params = [Omeg, eps]
#
#
#coeffs = Model(delta_arr, params)
#np.random.seed(53)
#sig = 0.03
#noise = np.random.normal(0, sig, (len(delta_arr), 3))
#data = coeffs + noise
#
#dd = 100
#d_arr = np.arange(-dmax, dmax + dd, dd)
#
#guess = [23000, 0]
#initial = Model(d_arr, guess)
#plt.figure()
#plt.plot(d_arr, initial, lw=3)
#plt.scatter(delta_arr, data[:,0], label="0")
#plt.scatter(delta_arr, data[:,1], label="1")
#plt.scatter(delta_arr, data[:,2], label="2")
#plt.legend()
#plt.title('Initial Guess')
#plt.show()

#################   PROCESS REAL DATA   #########################

guess = [20, 0]
initial = Model(b, guess)

plt.figure()
plt.plot(b, initial, lw=3)
plt.scatter(bias, data[:,2], label="2")
plt.scatter(bias, data[:,1], label="1")
plt.scatter(bias, data[:,0], label="0")
plt.legend()
plt.title('Initial Guess')
plt.show()

#%%
##################   FIT TO DATA   ###########################

params = Parameters()
params.add('Omeg', value=guess[0], min=0, max=40)
params.add('eps', value=guess[1], vary=True)


result = minimize(Residuals, params, method='nelder', args=(bias, data))
report_fit(result)



final = Model(b, result.params)
plt.figure()
plt.plot(b, final, lw=3)
plt.scatter(bias, data[:,2], label="0")
plt.scatter(bias, data[:,1], label="1")
plt.scatter(bias, data[:,0], label="2")
plt.legend()
plt.title('Fitted Parameters')
plt.xlabel('Detuning (kHz)')
plt.ylabel('Population')
plt.show()