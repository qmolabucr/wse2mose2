'''
figS45.py

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
A visualization script to display supplementary Fig. S4.5 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import inset_locator

from scipy.ndimage.filters import gaussian_filter
from os.path import join

from imaging_datasets import load_spectro_image_dataset

import qmodisplay as display
from qmomath import h_eV, c_nm, lorentzian, lorentzian_fit

# c = [x1, x2, y1, y2]
def avg_region(c, data):
    return np.mean(data[c[2]:c[3],c[0]:c[1]])
#

def lambda_Vg_phase(ax1, ax2, ax1cb, dataset, v1=0, v2=0.4):
    Vsd, Vg, w, pw, d = dataset
    rows, cols, Mw, Mv = d.shape
    coords = [38, 58, 26, 40]
    ix1 = np.searchsorted(Vg, v1)
    ix2 = np.searchsorted(Vg, v2)

    I = np.zeros((Mw, Mv))
    for i in range(Mw):
        for j in range(Mv):
            I[i,j] = avg_region(coords, d[:,:,i,j]/pw[i])
    #

    pc = gaussian_filter(np.abs(I), 1.0)
    cmap, cNorm, scalarMap = display.colorscale_map(pc, cmin=0, cmax=1, mapname='viridis')
    display.change_axes_colors(ax1cb, 'w')
    mpl.rcParams['axes.edgecolor'] = 'w'
    cb1 = mpl.colorbar.ColorbarBase(ax1cb, cmap=cmap, norm=cNorm, orientation='vertical')
    cb1.set_label(r"responsivity" + '\n' + r"(nA/mW)", labelpad=5)

    display.yaxis_right(ax2)

    ax1.imshow(pc, extent=[np.min(Vg), np.max(Vg), np.max(w), np.min(w)], cmap='viridis', interpolation='none', aspect='auto')
    ax1.set_ylabel(r"wavelength (nm)")
    ax1.set_xlabel(r"V$_{G}$ (V)")
    ax1.axvline(v1, color='r', ls='--')
    ax1.axvline(v2, color='r', ls='--')

    E = np.zeros(Mw)
    for i in range(Mw):
        E[i] = (h_eV*c_nm)/w[i]
    Ie = np.mean(I[:,ix1:ix2],axis=1)
    Ie = gaussian_filter(np.abs(Ie), 1.0)
    p, perr = lorentzian_fit(E, Ie)
    ftx = np.linspace(np.min(E), np.max(E), 200)
    ft = lorentzian(ftx, *p)

    ax2.plot(E, Ie, 'o', color='k')
    ax2.plot(ftx, ft, color='grey')
    ax2.set_xlabel(r"photon energy (eV)")

    ytks = np.arange(0.55, 0.85, 0.05)
    ax2.set_yticks(ytks)
    ytklbls = ['0.55 ','0.60 ', '0.65 ', '0.70 ', '0.75 ', '0.80 ']
    ax2.set_yticklabels(ytklbls)
    ax2.set_ylabel(r"responsivity (nA/mW)")
# lambda_Vg_phase

def tile_plot(ax1, dataset, spinewidth=0.2):
    Vsd, Vg, w, pw, d = dataset
    rows, cols, Mw, Mg = d.shape
    cmap, cnorm, smap = display.colorscale_map(d, mapname='PuOr_r', cmin=-10, cmax=10)

    rcoords = np.array(range(0,Mw,1))
    ccoords = np.array(range(0,Mg,1))

    width = 1.0/len(ccoords)
    height = 1.0/len(rcoords)
    rix = 0
    for i in rcoords:
        cix = 0
        for j in ccoords:
            ax = plt.axes([i, j, 9876, 54321], zorder=10)
            pos = inset_locator.InsetPosition(parent=ax1, lbwh=[cix*width, rix*height, width, height])
            ax.set_axes_locator(locator=pos)
            for k in ax.spines:
                ax.spines[k].set_linewidth(spinewidth)
            ax.imshow(d[:,:,i,j], cmap=cmap, norm=cnorm, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            cix += 1
        rix += 1
    ax1.set_ylim(np.min(w), np.max(w))
    ax1.set_xlim(np.min(Vg), np.max(Vg))

    ax1.set_yticks(np.arange(1175,1325+1,25))
    ax1.set_ylabel(r'laser wavelength (nm)')
    ax1.set_xticks([-1, -0.5, 0.0, 0.5, 1.0])
    ax1.set_xlabel(r'V$_{G}$ (V)')
    #ax1.set_title(r'V$_{SD}$ = ' + str(round(Vsd[0]*1e-3,2)) + ' V')
# end tile_plot

if __name__ == '__main__':
    save, svfile = display.argsave()

    dataset = load_spectro_image_dataset()
    rows, cols, Mw, Mg = dataset[4].shape

    display.paper_figure_format(labelpad=3)

    yinches = 6.5
    xinches = 6.5
    fi = display.figure_inches(__file__, xinches, yinches)
    fig1 = fi.get_fig()
    xmargin = 0.95
    ymargin = 0.60

    xint = 0.14
    yint = 0.65
    width = 2.25
    width2 = 2*width+xint
    height2 = (Mw/Mg)*(rows/cols)*width2

    inheight = 0.35*width
    inwidth = 0.05*width
    inmargin = 0.035*width

    ax1 = fi.make_axes([xmargin, ymargin+width+yint, width2, height2])
    ax2 = fi.make_axes([xmargin, ymargin, width, width])
    ax2cb = fi.make_axes([xmargin+inmargin, ymargin+width-inmargin-inheight, inwidth, inheight],zorder=999)
    ax3 = fi.make_axes([xmargin+width+xint, ymargin, width, width])

    tile_plot(ax1, dataset)

    lambda_Vg_phase(ax2, ax3, ax2cb, dataset)

    x1 = 0.12
    y1 = 0.96
    y2 = 0.46
    lblparams = {'fontsize':16, 'weight':'bold'}
    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x1, y2, 'b', **lblparams)

    if save:
        plt.savefig(join(svfile, 'figS45.png'), dpi=300)

    plt.show()
