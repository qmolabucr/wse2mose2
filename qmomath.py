'''
qmomath.py

version 1.1
last updated: December 2020

by Trevor Arp
Quantum Materials Optoelectronics Laboratory
Department of Physics and Astronomy
University of California, Riverside, USA

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Description:
This is generic calculation and analysis code, modified from the QMO Lab's internal code
libraries, as such all or part of this code may be published elsewhere at some point in
the future.

See accompanying README.txt for instructions on using this code.
'''

import numpy as np
import warnings

from scipy.optimize import curve_fit
from scipy.fftpack import fft, fftfreq

h_eV = 4.135667662e-15 # eV s
c_nm = 2.99792458e17 # nm/s (speed of light)
kb_eV = 8.6173324e-5 # eV/K

def gauss(x, A, x0, sigma):
	'''
	A one-dimensional Gaussian function given by

	y = A*Exp[-(x-x0)^2/(2*sigma^2)]

	Args:
		x : the independent variable
		A (float) : The Amplitude
		x0 (float) : The center x-value
		sigma (float) : The standard deviation
	'''
	return A*np.exp(-(x-x0)**2/(2*sigma**2))
# end gauss

def lorentzian(x, A, x0, G, y0):
	'''
	A basic normalized Lorentzian function
	               0.5*G
	y(x) = A*---------------------
	         (x-x0)^2 + (0.5*G)^2

	Args:
		x : the independent variable
		A (float) : The Amplitude
		x0 (float) : The center x-value
		G (float) : The full width at half maximum
		y0 (float) : The offset
	'''
	return A*0.5*G/((x-x0)**2 + (0.5*G)**2) + y0
# end lorentzian

def dydx(x, y):
	'''
	A simple numerical derivative using numpy.gradient, computes using 2nd order central differences.
    Certain data may require more thoughtful means of differentiation.

	Args:
		x : The independent variable
		y : The dependent variable, should have same length as x

	Returns:
		A simple numerical derivative, more complex differentiation may be required sometimes.
	'''
	return np.gradient(y)/np.gradient(x)
# end dydx

def generic_fit(x, y, p0, fitfunc, warn=True, maxfev=2000):
    '''
    A generic wrapper for curve_fit, accepts data (x,y), a set of initial parameters p0 and a function
    to fit to.

    Args:
        x : The independent variable
        y : The dependent variable
        p0 : The initial parameters for the fitting function
        fitfunc : The function to fit
        warn (bool, optional) : If True (default) will print error message when fitting
        maxfev (int, optional) : The 'maxfev' parameter of scipy.optimize.curve_fit

    Returns:
        p, perr

            p - The fitting parameters that optimize the fit, or the initial parameters if the fit failed.

            perr - Estimated error in the parameters, or zero if the fit failed.
    '''
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, plconv = curve_fit(fitfunc, x, y, p0=p0, maxfev=maxfev)
            perr = np.sqrt(np.abs(np.diag(plconv)))
    except Exception as e:
        p = p0
        perr = [0 for _ in range(len(p0))]
        if warn:
            print("Error fitting.generic_fit: Could not fit, parameters set to initial")
            print(str(e))
    return p, perr
# end generic_fit

def lorentzian_fit(x, y, p0=None, warn=True):
    '''
    Fits data to a Lorentzian function defined by math.lorentzian

    Returns the fit parameters and the errors in the fit parameters as (p, perr)

    Args:
        x : The independent variable
        y : The dependent variable
        p0 (optional): The initial parameters for the fitting function. If None (default) will estimate starting parameters from data
        warn (bool, optional) : If True (default) will print error message when fitting fails.

    Returns:
        p, perr

            p - The fitting parameters that optimize the fit, or the initial parameters if the fit failed.

            perr - Estimated error in the parameters, or zero if the fit failed.
    '''
    l = len(y)
    if len(x) != l :
        print("Error fitting.gauss_fit: X and Y data must have the same length")
        return
    if p0 is None:
        a = np.max(y) - np.mean(y)
        x0 = np.mean(x)
        y0 = np.mean(y)
        p0=(a, x0, 0.5, y0)
    return generic_fit(x, y, p0, lorentzian, warn=warn)
# end gauss_fit

def normfft_freq(t, d):
    '''
    Calculates a normalized Fast Fourier Transform (FFT) of the given data and the frequency samples for
    a given an (evenly sampled) time series

    Args:
        t : An evenly sampled times series for the data
        d : The data

    Returns:
        A tuple containing (freq, fft)

            freq - The sampling frequencies for the FFT, based on the argument t.

            fft - The normalized Fast Fourier Transform of the data.
    '''
    n = len(d)
    f = fft(d)
    f = 2.0*np.abs(f)/n
    freq = fftfreq(n, d=np.mean(np.diff(t)))
    return freq, f
# end normfft

if __name__ == "__init__":
    print("Calculation code from the QMO Lab")
