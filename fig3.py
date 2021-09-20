'''
fig3.py

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
A visualization script to display Fig. 4 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np
from numpy.fft import fft2, fftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib import rcParams

from os.path import join

import qmodisplay as display
from qmomath import dydx, normfft_freq, leastsq_2D_fit
from imaging_datasets import load_image_dataset

def plot_2deriv_map(ax, axtop, axcb, dataset, alpha=1.820, eta=2.5):
    Vsd, Vg, d, pc = dataset
    rows, cols = pc.shape
    V = 1e3*Vsd/(alpha*eta) # Convert to meV

    dpc = np.zeros(pc.shape)
    d2pc = np.zeros(pc.shape)
    for i in range(cols):
        dpc[:,i] = dydx(V, pc[:,i])
        d2pc[:,i] = dydx(V, dpc[:,i])
    #
    d2pc = d2pc*1e3 # convert to pA/meV^2

    rcParams['axes.edgecolor'] = 'w'
    display.change_axes_colors(axcb, 'w')
    cmap, cnorm, smap = display.colorscale_map(d2pc, cmin=-4, cmax=8, mapname='inferno') #, truncate=(0.0,0.9))
    #cmap, cnorm, smap = display.colorscale_map(d2pc, mapname='inferno')
    display.make_colorbar(axcb, cmap, cnorm, ticks=np.arange(-4, 8.1,4), orientation='horizontal')
    axcb.set_xlabel('d$^{2} I_{PC}$/d$\Delta E^{2}$ (pA/meV$^{2}$)    ', labelpad=1)

    # To get Vg on the y-axis with the right orientation rotate it 90 degrees then flip vertically
    d2pc = np.rot90(d2pc)
    ax.imshow(np.flipud(d2pc), extent=[np.min(V), np.max(V), np.max(Vg), np.min(Vg)], cmap=cmap, norm=cnorm, interpolation='bilinear', aspect='auto')


    x0 = -99
    ax.annotate('', xy=(x0, 0.9), xytext=(x0-30, 0.9), arrowprops=dict(arrowstyle="<|-|>", color='w'), annotation_clip=False)
    ax.annotate('', xy=(x0, 0.82), xytext=(x0, 0.98), arrowprops=dict(arrowstyle="-", color='w'), annotation_clip=False)
    ax.annotate('', xy=(x0-30, 0.82), xytext=(x0-30, 0.98), arrowprops=dict(arrowstyle="-", color='w'), annotation_clip=False)
    ax.annotate('30 meV', xy=(x0-15, 0.77), xytext=(x0-15, 0.77), annotation_clip=False, ha='center', color='w')

    ax.text(0.7, 0.7, r'$\eta = \dfrac{t}{d_{\perp}} \approx $'+str(eta), transform=ax.transAxes,  ha='center', color='w')

    axtop.set_xlim(np.min(Vsd), np.max(Vsd))

    ax.set_xlim(np.min(V), np.max(V))
    ax.set_xticks([-150, -100, -50, 0])
    ax.set_xlabel(r'$\Delta E =  eV_{SD}/\alpha\eta$ (meV)')

    ax.set_ylim(np.max(Vg), np.min(Vg))
    ax.set_yticks([-1, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
    ax.set_ylabel(r'V$_{G}$ (V)', labelpad=-2)
# end plot_phase_map

def plot_avg_d2pc(ax, axtop, dataset, alpha=1.820, eta=2.5, Vg1=-0.3, Vg2=0.4):
    Vsd, Vg, d, phase = dataset
    ix1 = np.searchsorted(Vg, Vg1)
    ix2 = np.searchsorted(Vg, Vg2)
    ixs = np.searchsorted(Vsd, -0.723)
    Vsd = Vsd[ixs:]
    pc = np.mean(phase[ixs:,ix1:ix2], axis=1)

    DE = 1e3*Vsd/(alpha*eta) # convert to meV
    dpc = dydx(DE, pc)
    d2pc = dydx(DE, dpc)
    ax.plot(DE, 1e3*d2pc, color='C2')
    ax.set_xlabel(r'$\Delta E = eV_{\mathrm{SD}}/\alpha\eta$ (meV)')
    ax.set_ylabel(r'd$^{2} I_{\mathrm{PC}}$/d$ \Delta E^{2}$ (pA/meV$^{2}$)', labelpad=-4)
    ax.set_xlim(np.min(DE), np.max(DE))
    ax.set_ylim(-10, 3)
    axtop.set_xlim(np.min(Vsd), np.max(Vsd))
# end plot_avg_d2pc

def plot_avg_d2pc_fourier(ax, dataset, alpha=1.820, eta=2.5, Vg1=-0.3, Vg2=0.4):
    Vsd, Vg, d, phase = dataset
    ix1 = np.searchsorted(Vg, Vg1)
    ix2 = np.searchsorted(Vg, Vg2)
    ixs = np.searchsorted(Vsd, -0.723)
    Vsd = Vsd[ixs:]
    pc = np.mean(phase[ixs:,ix1:ix2], axis=1)

    V = 1e3*Vsd/(alpha*eta)
    dpc = dydx(V, pc)
    d2pc = dydx(V, dpc)
    f, fft = normfft_freq(V, d2pc)
    N = len(f)
    ixc = N//2
    f = np.abs(f[1:ixc])
    fft = np.abs(fft[1:ixc])
    fft = fft/np.max(fft) # normalize

    display.xaxis_top(ax)
    ax.tick_params(pad=0)

    ax.plot(f, 9*fft, color='C2')
    ax.axvline(1/28, ls='--', color='C0')
    ax.text(0.40, 0.85, "1/(28 meV)", ha='left', color='C0', transform=ax.transAxes)

    ax.set_ylim(0, 9.5)
    ax.set_xlim(0.0, 0.1)

    ax.set_xticks([0, 0.05, 0.1])
    ax.set_xticklabels(['0.0', '0.05', '0.1'])
    ax.tick_params(axis='y', pad=3)

    ax.set_xlabel(r"$\Delta E$ Fourier"+'\n'+r"component (meV$^{-1}$)", linespacing=0.8)
    ax.set_ylabel("spectral density (arb.)", labelpad=3)
# end plot_avg_fourier

def plot_dpc_FFT(ax, axcb, dataset, c='w'):
    Vsd, Vg, d, pc = dataset
    rows, cols = pc.shape

    V = Vsd
    pc = np.flipud(pc)
    dpc = np.zeros(pc.shape)
    for j in range(cols):
        dpc[:,j] = dydx(V, pc[:,j])
    #

    dpc_fft = np.abs(fftshift(fft2(dpc)))
    dpc_fft = 1.111*dpc_fft/np.max(dpc_fft)

    frows = fftfreq(rows, d=np.mean(np.diff(V)))
    fcols = fftfreq(cols, d=np.mean(np.diff(Vg)))

    fcmap, fcnorm, fcsmap = display.colorscale_map(dpc_fft, cmin=0, cmax=0.1, mapname='inferno')
    rcParams['axes.edgecolor'] = c
    display.make_colorbar(axcb, fcmap, fcnorm, orientation='vertical')
    display.change_axes_colors(axcb, c)
    axcb.set_ylabel(r'd$ I_{\mathrm{PC}}$/d$ \Delta E$ FT (arb.)', labelpad=3)

    exnt = [np.min(fcols), np.max(fcols), np.min(frows), np.max(frows)]
    ax.imshow(dpc_fft, cmap=fcmap, norm=fcnorm, extent=exnt, aspect='auto', interpolation='none')

    ax.axhline(1/0.133, color='C0', ls='--')
    ax.text(0.0, 0.39, r"(133 mV)$^{-1}$", ha='left', color='C0', fontsize=10, transform=ax.transAxes)

    display.yaxis_right(ax)
    ax.set_ylim(0, 20)
    ax.set_yticks(np.arange(0,21,5))
    ax.set_ylabel(r"$V_{\mathrm{SD}}$ Fourier component (V$^{-1}$)")
    ax.set_xlabel(r'$V_{\mathrm{G}}$ Fourier component (V$^{-1}$)', labelpad=1)
# end plot_d2pc_FFT

def fit_FFT_peak(x,y,d):
    def gauss2D(x, y, A, x0, sigmax, y0, sigmay):
        '''
        A two-dimensional Gaussian function given by

        f(x,y) = A*Exp[ -(x-x0)^2/(2*sigmax^2) - (y-y0)^2/(2*sigmay^2) ]

        Args:
        	A (float) : The Amplitude
        	x0 (float) : The center x-value
        	sigmax (float) : The standard deviation in the x direction
        	sigmay (float) : The standard deviation in the y direction
        '''
        out = np.zeros((len(y), len(x)))
        for i in range(len(y)):
            for j in range(len(x)):
                out[i,j] = A*np.exp(-(x[j]-x0)**2 / (2*sigmax**2) - (y[i]-y0)**2 / (2*sigmay**2))
        return out
    # end gauss2D

    # Take only ht epositive half of the FFT
    N = len(y)
    x = fftshift(x)
    y = y[:N//2]
    y = y[::-1]
    d = d[:N//2]

    p0 = (1.0, 0.0, 2, 30, 10)
    p, perr = leastsq_2D_fit(x, y, d, p0, gauss2D)
    ft = gauss2D(x, y, *p)
    # print(p)
    # print(perr)
    return p, perr, (x, y, ft)
#

def plot_d2pc_FFT(ax, axcb, dataset, alpha=1.820, eta=2.5, c='w'):
    Vsd, Vg, d, pc = dataset
    rows, cols = pc.shape

    V = Vsd/(alpha*eta)
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
    axcb.set_ylabel(r'd$^{2} I_{\mathrm{PC}}$/d$ \Delta E^{2}$ FT (arb.)', labelpad=1)

    exnt = [np.min(fcols), np.max(fcols), np.min(frows), np.max(frows)]
    ax.imshow(d2pc_fft, cmap=fcmap, norm=fcnorm, extent=exnt, aspect='auto', interpolation='none')

    p, perr, (x, y, ft) = fit_FFT_peak(fcols, frows[1:], d2pc_fft[1:,:])
    ax.axhline(p[3], color='C0', ls='--')
    val = int(1000.0/p[3])
    ax.text(0.0, 0.40, r"("+str(val)+" meV)$^{-1}$", ha='left', color='C0', fontsize=10, transform=ax.transAxes)

    display.yaxis_right(ax)
    ax.set_ylim(0, 87)
    ax.set_ylabel(r"$\Delta E$ Fourier component (eV$^{-1}$)")
    ax.set_xlabel(r'$V_{\mathrm{G}}$ Fourier component (V$^{-1}$)', labelpad=1)
# end plot_d2pc_FFT

def plot_deriv_map(ax, axcb, dataset):
    Vsd, Vg, d, pc = dataset
    rows, cols = pc.shape

    dpc = np.zeros(pc.shape)
    for i in range(cols):
        dpc[:,i] = dydx(Vsd, pc[:,i]) # units pA/meV

    rcParams['axes.edgecolor'] = 'w'
    display.change_axes_colors(axcb, 'w')
    cmap, cnorm, smap = display.colorscale_map(dpc,  mapname='plasma')
    display.make_colorbar(axcb, cmap, cnorm, ticks=np.arange(-15,31,15), orientation='horizontal')
    axcb.set_xlabel('d$I_{PC}$/d$V_{\mathrm{SD}}$ (nA/V)', labelpad=1)

    # To get Vg on the y-axis with the right orientation rotate it 90 degrees then flip vertically
    dpc = np.rot90(dpc)
    ax.imshow(np.flipud(dpc), extent=[np.min(Vsd), np.max(Vsd), np.max(Vg), np.min(Vg)], cmap=cmap, norm=cnorm, interpolation='bilinear', aspect='auto')
    ax.set_xlim(np.min(Vsd), np.max(Vsd))
    ax.set_xlabel(r'$V_{\mathrm{SD}}$ (V)', labelpad=1)

    ax.set_ylim(np.max(Vg), np.min(Vg))
    ax.set_yticks([-1, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
    ax.set_ylabel(r'V$_{G}$ (V)', labelpad=3)
# end plot_deriv_map

def plot_avg_dpc(ax,  dataset, Vg1=-0.3, Vg2=0.4):
    Vsd, Vg, d, phase = dataset
    ix1 = np.searchsorted(Vg, Vg1)
    ix2 = np.searchsorted(Vg, Vg2)
    ixs = np.searchsorted(Vsd, -0.723)
    Vsd = Vsd[ixs:]
    pc = np.mean(phase[ixs:,ix1:ix2], axis=1)
    dpc = dydx(Vsd, pc)

    ax.plot(Vsd, dpc, color='C1')
    ax.set_yticks(np.arange(2,19,4))
    ax.set_xlim(np.min(Vsd), np.max(Vsd))
    ax.set_xlabel(r'$V_{\mathrm{SD}}$ (V)', labelpad=2)
    ax.set_ylabel(r'd$I_{\mathrm{PC}}$/d$ V_{\mathrm{SD}}$ (nA/V)')
# end plot_avg_d2pc

if __name__ == '__main__':
    save, svfile = display.argsave()

    dataset = load_image_dataset()

    display.paper_figure_format(fntsize=10, labelpad=3)

    xinches = 9.85
    yinches = 7.4
    fi = display.figure_inches(__file__, xinches, yinches)
    fig1 = fi.get_fig()
    xmargin = 0.7
    ymargin = 0.5
    height = 3.0 #3.5
    width = height/2
    yint = 0.7
    xint = 0.4
    width2 = 3.2

    ystart = ymargin + height + yint

    xstart = xmargin
    cbwidth = 0.4*height
    cbheight = 0.08*width
    cbmargin = 0.04*width
    ax1, ax1top = fi.make_dualx_axes([xstart, ymargin, height, height])
    ax1cb = fi.make_axes([xstart+0.5*width2, ymargin+height-cbmargin-cbheight, cbwidth, cbheight], zorder=10)

    ax4 = fi.make_axes([xstart, ystart, height, height])
    ax4cb = fi.make_axes([xstart+0.5*width2, ystart+height-cbmargin-cbheight, cbwidth, cbheight])

    cbwidth = 0.07*width
    cbheight = 0.33*height
    inmargin = 0.03*width2
    inwidth = 1.5
    xstart = xstart + width2 + xint
    ax2, ax2top = fi.make_dualx_axes([xstart, ymargin, width2, height])
    ax2in = fi.make_axes([xstart+width2-inmargin-inwidth, ymargin+0.8*inmargin, inwidth, inwidth], zorder=2)

    ax5 = fi.make_axes([xstart, ystart, width2, height])

    xstart = xstart + width2 + xint/2
    ax3 = fi.make_axes([xstart, ymargin, width, height])
    ax3cb = fi.make_axes([xstart+cbmargin, ymargin+height-cbheight-2.5*cbmargin, cbwidth, cbheight])

    ax6 = fi.make_axes([xstart, ystart, width, height])
    ax6cb = fi.make_axes([xstart+cbmargin, ystart+height-cbheight-2.5*cbmargin, cbwidth, cbheight])

    # Bottom Row
    plot_2deriv_map(ax1, ax1top, ax1cb, dataset)
    plot_avg_d2pc(ax2, ax2top, dataset)
    plot_avg_d2pc_fourier(ax2in, dataset)
    plot_d2pc_FFT(ax3, ax3cb, dataset)

    # Top Row
    plot_deriv_map(ax4, ax4cb, dataset)
    plot_avg_dpc(ax5, dataset)
    plot_dpc_FFT(ax6, ax6cb, dataset)


    x1 = 0.02
    x2 = 0.40
    x3 = 0.766
    y1 = 0.96
    y2 = 0.46
    lblparams = {'fontsize':16, 'weight':'bold'}
    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x2, y1, 'b', **lblparams)
    plt.figtext(x3, y1, 'c', **lblparams)
    plt.figtext(x1, y2, 'd', **lblparams)
    plt.figtext(x2, y2, 'e', **lblparams)
    plt.figtext(x3, y2, 'f', **lblparams)

    if save:
        plt.savefig(join(svfile, 'fig3.png'), dpi=300)

    plt.show()
