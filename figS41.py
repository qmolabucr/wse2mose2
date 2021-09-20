'''
figS41.py

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
A visualization script to display supplementary Fig. S4.1 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch as FA

from os.path import join

import qmodisplay as display
from qmomath import generic_fit, lorentzian

def load_intralayer_data(fl='intra-800-900', pc0=-0.810, N=100):
    dir = join('datasets','intralayer-exciton')
    Vg = np.loadtxt(join(dir, fl+'-x.txt'))
    Eph = np.loadtxt(join(dir, fl+'-y.txt'))
    pc = np.loadtxt(join(dir, fl+'-d.txt'))
    pc = 100*np.abs(pc-pc0) # make positive to deal with sign flip and convert to picoamps
    Eph = Eph[::-1]
    pc = pc[::-1, :]
    return Eph, Vg, pc
# load_data

def triplepeak(x, *pm):
    return lorentzian(x, pm[0], pm[1], pm[2], 0) + lorentzian(x, pm[3], pm[4], pm[5], 0) + lorentzian(x, pm[6], pm[7], pm[8], 0)
# end triplepeak

def calc_redshift(dataset, Vgcut=-5):
    Eph, Vg, pc = dataset
    rows, cols = pc.shape
    ixc = np.searchsorted(Vg, Vgcut)
    N = ixc
    p0 = [0.2, 1.44, 0.025, 0.2, 1.489, 0.025, 0.2, 1.524, 0.025]
    peaks = np.zeros((N, 6))
    redshift = np.zeros((N, 6))
    for j in range(N):
        p, perr = generic_fit(Eph, pc[:,j], p0, triplepeak)
        peaks[j,0] = p[7]
        peaks[j,1] = p[4]
        peaks[j,2] = p[1]
        peaks[j,3] = perr[7]
        peaks[j,4] = perr[4]
        peaks[j,5] = perr[1]
        if j != 0:
            p0 = p
    #

    for k in range(3):
        plin = np.polyfit(Vg[:N], peaks[:,k],1)
        redshift[:,k] = peaks[:,k] - np.polyval(plin, 0)
        redshift[:,3+k] = peaks[:,3+k]
    redshift = redshift*1e3
    return Vg[:N], peaks, redshift
# end calc_redshift

def show_pc_map(ax, axcb, dataset, Vgpk, peaks, c='w'):
    Eph, Vg, pc = dataset

    pcmap, pcnorm, pcsmap = display.colorscale_map(pc, cmax=20, mapname='inferno')
    rcParams['axes.edgecolor'] = c
    display.make_colorbar(axcb, pcmap, pcnorm, orientation='horizontal', ticks=np.arange(0,31,10))
    display.change_axes_colors(axcb, c)
    axcb.set_title('$pc$ (pA)', color=c)

    ax.imshow(np.flipud(pc), cmap=pcmap, norm=pcnorm, extent=[np.min(Vg), np.max(Vg), np.min(Eph), np.max(Eph)], aspect='auto', interpolation='none')
    ax.set_xlim(-20,20)
    ax.set_ylabel(r'$E_{\mathrm{PH}}$ (eV)')
    ax.set_xlabel(r'$V_{\mathrm{G}}$ (V)')

    aparams = {'color':c, 'lw':'1.2', 'ls':'-', 'arrowstyle':'-|>','mutation_scale':10}
    ln1 = np.polyfit(Vgpk, peaks[:,0],1)
    ax.add_patch(FA((-20, np.polyval(ln1, -20)),(3, np.polyval(ln1, 3)), **aparams))
    ax.text(3, 1.524, r"X$^{0}_{\mathrm{Mo}}$", color=c, va='center')

    ln2 = np.polyfit(Vgpk, peaks[:,1],1)
    ax.add_patch(FA((-20, np.polyval(ln2, -20)),(3, np.polyval(ln2, 3)), **aparams))
    ax.text(3, 1.492, r"X$^{+}_{\mathrm{Mo}}$", color=c, va='center')

    ln3 = np.polyfit(Vgpk, peaks[:,2],1)
    ax.add_patch(FA((-20, np.polyval(ln3, -20)),(3, np.polyval(ln3, 3)), **aparams))
    ax.text(3, 1.454, r"X$^{+}_{\mathrm{W}}$", color=c, va='center')
# end show_pc_map

def show_Eph_line_cuts(ax, dataset, Vgcut=20):
    Eph, Vg, pc = dataset
    rows, cols = pc.shape
    ixV = np.searchsorted(Vg, Vgcut)
    pcmap, pcnorm, pcsmap = display.colorscale_map(Vg, mapname='twilight', truncate=(0.2,0.8))
    txtprops = {'va':'center', 'ha':'left', 'transform':ax.transAxes}
    ax.text(0.03, 0.955, r"$V_{\mathrm{G}}=$", **txtprops)
    for i in range(0, ixV, 5):
        cval = pcsmap.to_rgba(Vg[i])
        ax.plot(Eph, pc[:,i], color=cval)
        lbl = str(round(Vg[i],1))+' V'
        if Vg[i] > 0:
            lbl = ' ' + lbl
        ax.text(0.03, 0.885-i*0.015, lbl, color=cval, **txtprops)
    ax.set_xlim(np.min(Eph), np.max(Eph))
    ax.set_ylim(0,25)
    ax.set_xlabel(r'$E_{\mathrm{PH}}$ (eV)')
    ax.set_ylabel('$pc$ (pA)')

    aparams = {'lw':'2', 'ls':'-', 'arrowstyle':'-|>','mutation_scale':15}
    ax.add_patch(FA((1.44, 9),(1.44, 6), color='#0c13cf', **aparams))
    ax.text(1.44-0.025, 6.5, r"X$^{+}_{\mathrm{W}}$")

    ax.add_patch(FA((1.489, 17),(1.489, 14), color='#15c462', **aparams))
    ax.text(1.489-0.025, 14.5, r"X$^{+}_{\mathrm{Mo}}$")

    ax.add_patch(FA((1.524, 25),(1.524, 22), color='#0c8140', **aparams))
    ax.text(1.524-0.025, 22.5, r"X$^{0}_{\mathrm{Mo}}$")
# end show_Eph_line_cuts

def show_redshift(ax, Vg, redshift):
    _Vg = [-20, 0]

    ax.errorbar(Vg, redshift[:,0], redshift[:,3], fmt='o', capsize=2, color='#0c8140', label=r"X$^{0}_{\mathrm{Mo}}$")
    ax.plot(_Vg, np.polyval(np.polyfit(Vg, redshift[:,0],1), _Vg), '--', color='#0c8140')
    ax.text(-19.7, redshift[0,0]+1, r"X$^{0}_{\mathrm{Mo}}$", color='k')

    ax.errorbar(Vg, redshift[:,1], redshift[:,4], fmt='o', capsize=2, color='#15c462', label=r"X$^{+}_{\mathrm{Mo}}$")
    ax.plot(_Vg, np.polyval(np.polyfit(Vg, redshift[:,1],1), _Vg), '--', color='#15c462')
    ax.text(-19.7, redshift[0,1]+2, r"X$^{+}_{\mathrm{Mo}}$", color='k')

    ax.errorbar(Vg, redshift[:,2], redshift[:,5], fmt='o', capsize=2, color='#0c13cf', label=r"X$^{+}_{\mathrm{W}}$")
    ax.plot(_Vg, np.polyval(np.polyfit(Vg, redshift[:,2],1), _Vg), '--', color='#0c13cf')
    ax.text(-19.7, redshift[0,2]+2, r"X$^{+}_{\mathrm{W}}$", color='k')

    #ax.legend()
    ax.set_ylabel(r'redshift (meV)')
    ax.set_xlabel(r'$V_{\mathrm{G}}$ (V)')
    ax.set_xlim(-20, 0)
# end show_redshift

if __name__ == '__main__':
    save, svfile = display.argsave()

    dataset = load_intralayer_data()
    Vg, peaks, redshift = calc_redshift(dataset)

    display.paper_figure_format(labelpad=5)

    xinches = 10.0
    yinches = 3.25
    fi = display.figure_inches('figS41', xinches, yinches)
    fig1 = fi.get_fig()
    xmargin = 0.75
    ymargin = 0.55

    height = 2.5
    width = height
    xint = 0.7

    cbwidth = 0.38*width
    cbheight = 0.06*height
    cbmargin = 0.05*width

    ax1 = fi.make_axes([xmargin, ymargin, width, height])
    ax1cb = fi.make_axes([xmargin+1.5*cbmargin, ymargin+cbheight+cbmargin, cbwidth, cbheight])
    ax2 = fi.make_axes([xmargin+xint+width, ymargin, width, height])
    ax3 = fi.make_axes([xmargin+2*xint+2*width, ymargin, width, height])

    show_pc_map(ax1, ax1cb, dataset, Vg, peaks)
    show_Eph_line_cuts(ax2, dataset)
    show_redshift(ax3, Vg, redshift)

    x1 = 0.01
    x2 = 0.34
    x3 = 0.66
    y1 = 0.93
    lblparams = {'fontsize':16, 'weight':'bold'}
    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x2, y1, 'b', **lblparams)
    plt.figtext(x3, y1, 'c', **lblparams)

    if save:
        plt.savefig(join(svfile, 'figS41.png'), dpi=300)

    plt.show()
