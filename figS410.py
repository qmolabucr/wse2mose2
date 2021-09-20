'''
figS410.py

version 1.2
last updated: September 2021

by Trevor Arp
Quantum Materials Optoelectronics Laboratory
Department of Physics and Astronomy
University of California, Riverside, USA

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Description:
A visualization script to display supplementary Fig. S4.10 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np

from os.path import join

import matplotlib.pyplot as plt

import qmodisplay as display
from qmomath import dydx, normfft_freq

from imaging_datasets import load_image_dataset

def get_photocurrent(dataset, Vg1=-0.3, Vg2=0.4):
    Vsd, Vg, d, phase = dataset
    rows, cols = phase.shape

    ix1 = np.searchsorted(Vg, Vg1)
    ix2 = np.searchsorted(Vg, Vg2)

    ixs = np.searchsorted(Vsd, -0.723)
    Vsd = Vsd[ixs:]
    pc = np.mean(phase[ixs:,ix1:ix2], axis=1)
    return pc, Vsd
# end plot_photocurrent

def plot_2deriv(ax, Vsd, pc, alpha=1.820, eta=2.5, dark=False):
    DE = 1e3*Vsd/(alpha*eta) # convert to meV
    dpc = dydx(DE, pc)
    d2pc = dydx(DE, dpc)
    #gl.display.yaxis_right(ax)
    if dark:
        ax.set_ylabel(r'd$^{2} I_{\mathrm{DARK}}$/d$ \Delta E^{2}$ (pA/meV$^{2}$)')
        clr = 'C1'
        ax.set_xlabel(r'$\Delta E = eV_{\mathrm{SD}}/\alpha\eta$ (meV)')
        ax.set_ylim(-2800, 5000)
    else:
        ax.set_ylabel(r'd$^{2} I_{\mathrm{PC}}$/d$ \Delta E^{2}$ (pA/meV$^{2}$)')
        clr='C2'
        ax.set_xticks([])
        ax.set_ylim(-3.25, 6.0)
    ax.plot(DE, 1e3*d2pc, color=clr)
    ax.set_xlim(-160, 30)
# end plot_2deriv

def plot_fourier(ax, Vsd, pc, alpha=1.820, eta=2.5, dark=False):
    V = Vsd/(alpha*eta)
    dpc = dydx(V, pc)
    d2pc = dydx(V, dpc)
    f, fft = normfft_freq(1e3*V, d2pc)
    N = len(f)
    ixc = N//2
    f = np.abs(f[1:ixc])
    fft = np.abs(fft[1:ixc])

    if dark:
        clr = 'C1'
        norm = 1e-4
        ax.set_xlim(0, 0.5)
    else:
        clr='C2'
        norm = 0.01
        ax.set_xlim(0, 0.1)
    #
    ax.plot(f, norm*fft, color=clr)
    ax.set_ylabel("spectral density")
    ax.set_xlabel(r"$1/ \Delta E$ (meV$^{-1}$)", labelpad=1)

    ax.axvline(1/30, ls='--', color='C0')
    ax.text(0.035, 0.9*np.max(norm*fft), "1/(30 meV)", ha='left', color='C0')
# end plot_fourier

def get_dark_current(rn='2019_05_16_10'):
    darkdir = join('datasets', 'dark current')
    d = np.loadtxt(join(darkdir, 'wide-dark-current.txt'))
    Vsd = np.loadtxt(join(darkdir, 'wide-dark-Vsd.txt'))
    return Vsd, d
# end get_dark_current

def convert(meV):
    return 1.0/(1e-3*meV)
# end convert

if __name__ == '__main__':
    save, svfile = display.argsave()

    dataset = load_image_dataset()

    display.paper_figure_format(labelpad=5)

    xmargin = 0.9
    ymargin = 0.55

    xint = 0.85
    yint = 0.25
    width = 3.5
    height = width
    width2 = 1.5

    xinches = 1.25*xmargin + width
    yinches = 1.5*ymargin + 2*height + yint
    fi = display.figure_inches('figS410', xinches, yinches)

    inmargin = 0.02*width

    ax1 = fi.make_axes([xmargin, ymargin, width, height])
    ax1in = fi.make_axes([xmargin+width-2.25*inmargin-width2, ymargin+height-inmargin-width2, width2, width2], zorder=2)

    ystart = ymargin + height + yint
    ax2 = fi.make_axes([xmargin, ystart, width, height])
    ax2in = fi.make_axes([xmargin+width-2.25*inmargin-width2, ystart+height-inmargin-width2, width2, width2], zorder=2)

    # [m,b] = get_Vgmax(dataset)
    # n = calc_dark_current() # Cark current shown in fig4

    _Vsd, dark = get_dark_current() # Dark current over a wider range of VSD
    pc, Vsd = get_photocurrent(dataset)

    plot_2deriv(ax1, _Vsd, dark, dark=True)
    plot_fourier(ax1in, _Vsd, dark, dark=True)

    plot_2deriv(ax2, Vsd, pc)
    plot_fourier(ax2in, Vsd, pc)

    lblparams = {'fontsize':16, 'weight':'bold'}
    x1 = 0.03
    y1 = 0.97
    y2 = 0.51
    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x1, y2, 'b', **lblparams)

    if save:
        plt.savefig(join(svfile, 'figS410.png'), dpi=300)

    plt.show()
