'''
fig4.py

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
A visualization script to display Fig. 4 from the paper 'Stacking enabled
strong coupling of atomic motion to interlayer excitons in van der Waals
heterojunction photodiodes'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np
from numpy.fft import fft2, fftshift, fftfreq

from scipy.interpolate import interp1d

from os.path import join

import matplotlib.pyplot as plt
from matplotlib import rcParams

import qmodisplay as display
from qmomath import dydx, generic_fit, normfft_freq

from imaging_datasets import load_image_dataset

def get_Vgmax(dataset, V1=-0.1):
    Vsd, Vg, d, pc = dataset
    rows, cols = pc.shape
    Vg_ = np.linspace(np.min(Vg), np.max(Vg),200)
    ix1 = np.searchsorted(Vsd, V1)
    Vgmax = np.zeros(rows)
    for i in range(ix1):
        Ifunc = interp1d(Vg, pc[i,:], kind='cubic')
        ixmax = np.argmax(Ifunc(Vg_))
        Vgmax[i] = Vg_[ixmax]
    p = np.polyfit(Vgmax[:ix1], Vsd[:ix1], 1)
    return p
# end find_Vg_of_max

def diode(V, Is, n, kT=0.02559):
    return Is*(np.exp(V/(n*kT))-1)
# end diode

def show_dark_current(ax):
    dir = join('datasets','dark current')
    Vsd = np.loadtxt(join(dir, 'dark-Vsd.txt'))
    d = np.loadtxt(join(dir, 'dark-current.txt'))

    ix1 = np.searchsorted(Vsd, -0.205)
    Vsd = Vsd[ix1:]
    d = d[ix1:]

    p0 = [12, 2.0]
    p, perr = generic_fit(Vsd, d, p0, diode)
    ft = diode(Vsd, p[0], p[1])

    ax.plot(Vsd, d, '.', color='darkblue', label='dark current')
    ax.plot(Vsd, ft, color='r', label='diode equation fit')
    ax.set_xlabel(r"$V_{\mathrm{SD}}$ (V)")
    ax.set_ylabel(r"dark current (nA)")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-30, 900)

    return p[1]
# end show_dark_current

def plot_photocurrent(axl, axr, dataset, Vg1=-0.3, Vg2=0.4):
    Vsd, Vg, d, phase = dataset
    rows, cols = phase.shape

    ix1 = np.searchsorted(Vg, Vg1)
    ix2 = np.searchsorted(Vg, Vg2)

    ixs = np.searchsorted(Vsd, -0.723)
    Vsd = Vsd[ixs:]
    pc = np.mean(phase[ixs:,ix1:ix2], axis=1)
    dpc = dydx(Vsd, pc)

    axl.plot(Vsd, pc, color='C4')
    axr.plot(Vsd, dpc, color='C1')
    axl.set_xlabel(r'$V_{\mathrm{SD}}$ (V)')
    axl.set_ylabel(r'photocurrent $I_{\mathrm{PC}}$ (nA)', labelpad=3)
    axr.set_ylabel(r'photoconductance'+'\n'+r' d$I_{\mathrm{PC}}$/d$V_{\mathrm{SD}}$ (nA/V)')
    for ax in [axr, axl]:
        ax.set_xlim(np.min(Vsd), np.max(Vsd))
    # ax.set_ylim(-4.5, 8.6)
    return pc, Vsd
# end plot_photocurrent

def plot_2deriv(ax, Vsd, pc, n=1.81, m=2.11):
    DE = 1e3*Vsd/(n*m) # convert to meV
    dpc = dydx(DE, pc)
    d2pc = dydx(DE, dpc)
    #display.yaxis_right(ax)
    ax.plot(DE, 1e3*d2pc, color='C2')
    ax.set_xlabel(r'$\Delta E = eV_{\mathrm{SD}}/\alpha\eta$ (meV)')
    ax.set_ylabel(r'd$^{2} I_{\mathrm{PC}}$/d$ \Delta E^{2}$ (pA/meV$^{2}$)')
    ax.set_xlim(np.min(DE), np.max(DE))
    ax.set_ylim(-5.5,2.8)
# end plot_2deriv

def plot_fourier(ax, Vsd, pc, n=1.81, m=2.11):
    V = Vsd/(n*m)
    dpc = dydx(V, pc)
    d2pc = dydx(V, dpc)
    f, fft = normfft_freq(1e3*V, d2pc)
    N = len(f)
    ixc = N//2
    f = np.abs(f[1:ixc])
    fft = np.abs(fft[1:ixc])

    display.xaxis_top(ax)
    ax.tick_params(pad=0)

    #ax.semilogx(f, fft, color='C2')
    ax.plot(f, 0.01*fft, color='C2')

    ax.axvline(1/30, ls='--', color='C0')
    #ax.axvline(1/22, ls='--', color='C2')
    ax.text(0.40, 0.85, "1/(30 meV)", ha='left', color='C0', transform=ax.transAxes)

    ax.set_ylim(0, 6.90)
    ax.set_xlim(0.0, 0.1)

    ax.set_xticks([0, 0.05, 0.1])
    ax.set_xticklabels(['0.0', '0.05', '0.1'])
    ax.tick_params(axis='y', pad=3)

    ax.set_xlabel(r"$\Delta E$ Fourier"+'\n'+r"component (meV$^{-1}$)", linespacing=0.8)
    ax.set_ylabel("spectral density (arb.)", labelpad=3)
# end plot_fourier

def convert(meV):
    return 1.0/(1e-3*meV)
# end convert

def plot_voltage_FFT(ax, axcb, dataset, n=1.820, m=2.228, c='w'):
    Vsd, Vg, d, pc = dataset
    rows, cols = pc.shape

    V = Vsd/(n*m)
    pc = np.flipud(pc)
    dpc = np.zeros(pc.shape)
    d2pc = np.zeros(pc.shape)
    for j in range(cols):
        dpc[:,j] = dydx(V, pc[:,j])
        d2pc[:,j] = dydx(V, dpc[:,j])
    #

    d2pc_fft = np.abs(fftshift(fft2(d2pc)))
    d2pc_fft = 1.111*d2pc_fft/np.max(d2pc_fft)

    frows = fftfreq(rows, d=np.mean(np.diff(V)))
    fcols = fftfreq(cols, d=np.mean(np.diff(Vg)))

    fcmap, fcnorm, fcsmap = display.colorscale_map(d2pc_fft, cmin=0, cmax=1, mapname='inferno')
    rcParams['axes.edgecolor'] = c
    display.make_colorbar(axcb, fcmap, fcnorm, orientation='vertical')
    display.change_axes_colors(axcb, c)
    axcb.set_ylabel(r'd$^{2} I_{\mathrm{PC}}$/d$ \Delta E^{2}$ FT (arb.)', labelpad=3)

    exnt = [np.min(fcols), np.max(fcols), np.min(frows), np.max(frows)]
    ax.imshow(d2pc_fft, cmap=fcmap, norm=fcnorm, extent=exnt, aspect='auto', interpolation='none')

    ax.axhline(1/0.030, color='C0', ls='--')
    #ax.axhline(1/30, color='C0', ls='--')
    ax.text(0.0, 0.33, r"(30 meV)$^{-1}$", ha='left', color='C0', fontsize=10, transform=ax.transAxes)

    # display.yaxis_right(ax)
    ax.set_ylim(10, 90)
    #ax.set_ylim(0.010, 0.090)
    ax.set_ylabel(r"$\Delta E$ Fourier component (eV$^{-1}$)")
    ax.set_xlabel(r'$V_{\mathrm{G}}$ Fourier component (V$^{-1}$)', labelpad=0)
# end plot_phase_map

if __name__ == '__main__':
    save, svfile = display.argsave()

    dataset = load_image_dataset()

    display.paper_figure_format(fntsize=10, labelpad=5)

    xmargin = 0.75
    ymargin = 0.55

    width2 = 1.5
    xint = 0.7
    yint = 0.85
    width = 3.25
    height = width

    xinches = 1.25*xmargin + 2*width + 2*xint + width2
    yinches = 4.0
    fi = display.figure_inches('fig4', xinches, yinches)

    inmargin = 0.04*width
    inmargin2 = 0.17*width
    ystart = ymargin
    ax11 = fi.make_axes([xmargin, ystart, width, height])
    ax11inl, ax11inr = fi.make_dualy_axes([xmargin+inmargin2, ystart+width-width2-inmargin, width2, width2], zorder=2, color_left='C4', color_right='C1')

    ax12 = fi.make_axes([xmargin+width+xint, ystart, width, height])
    ax12in = fi.make_axes([xmargin+2*width+xint-inmargin-width2, ystart+inmargin, width2, width2], zorder=2)

    cbwidth = 0.06*width2
    cbheight = 0.33*height
    cbmargin = 0.03*width2
    ax13 = fi.make_axes([xmargin+2*width+2*xint, ystart, width2, height])
    ax13cb = fi.make_axes([xmargin+2*width+2*xint+cbmargin, ystart+height-cbheight-3*cbmargin, cbwidth, cbheight])

    [eta,b] = get_Vgmax(dataset)
    alpha = show_dark_current(ax11)
    pc, Vsd = plot_photocurrent(ax11inl, ax11inr, dataset)

    plot_2deriv(ax12, Vsd, pc, n=alpha, m=eta)
    plot_fourier(ax12in, Vsd, pc, n=alpha, m=eta)
    plot_voltage_FFT(ax13, ax13cb, dataset)

    lblparams = {'fontsize':16, 'weight':'bold'}
    x1 = 0.04
    x2 = 0.42
    x3 = 0.79
    y1 = 0.94
    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x2, y1, 'b', **lblparams)
    plt.figtext(x3, y1, 'c', **lblparams)

    if save:
        plt.savefig(join(svfile, 'fig4.png'), dpi=300)

    plt.show()
