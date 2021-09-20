'''
figS46.py

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
A visualization script to display supplementary Fig. S4.6 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm

from os.path import join

from imaging_datasets import load_image_dataset
import qmodisplay as display

Vgcuts = False

def disp_pc_ref(axpc, axrf, data, points, cls):
    Vg, Vsd, d, r = data
    ixVg = np.searchsorted(Vg, 0.2)
    ixVsd = np.searchsorted(Vsd, -0.1)

    cmap = plt.get_cmap('PuOr')
    cNorm  = colors.Normalize(vmin=-7.5, vmax=7.5)
    axpc.imshow(d[:,:,ixVg, ixVsd], cmap=cmap, norm=cNorm)
    axrf.imshow(r[:,:,ixVg, ixVsd], cmap='Greys', vmin=0.4, vmax=0.6)
    for i in range(3):
        pt = points[i]
        axpc.plot(pt[0], pt[1], 'o', color=cls[i])
        axrf.plot(pt[0], pt[1], 'o', color=cls[i])
    for ax in [axrf, axpc]:
        ax.set_xticks([])
        ax.set_yticks([])
    axpc.set_title(r"V$_{G}$=0.2 V, V$_{SD}$=-0.1 V", fontsize=12)
    axrf.set_title("Reflection Image", fontsize=12)
# end disp_pc_ref

def phase_map_cuts(ax1, data, pt, cl, cmap, cnorm, ttl, showy=False):
    Vg, Vsd, d, r = data
    pci = gaussian_filter(d[pt[1], pt[0],:,:],1.0)
    rows, cols = pci.shape
    for spine in ax1.spines.values():
        spine.set_edgecolor(cl)
    ax1.imshow(np.rot90(pci), extent=[np.min(Vg), np.max(Vg), np.min(Vsd), np.max(Vsd)], cmap=cmap, norm=cnorm, aspect='auto', interpolation=None)
    ax1.set_xlabel(r"V$_{G}$ (V)")
    if showy:
        ax1.set_ylabel(r"V$_{SD}$ (V)")
    else:
        ax1.set_yticks([])
    ax1.set_title(ttl, color=cl, fontsize=12)
# end phase_map

if __name__ == '__main__':
    save, svfile = display.argsave()

    Vsd, Vg, d, pc, r = load_image_dataset("Vg_Vsd_narrow", refdata=True)
    rows, cols, Mg, Mb = d.shape
    N = Mg*Mb

    # # Convert to V
    # Vsd = Vsd*1e-3

    display.paper_figure_format(labelpad=5)

    xinches = 6.5
    yinches = 6.0
    fi = display.figure_inches('figS46', xinches, yinches)
    xmargin = 0.75
    ymargin = 0.5

    width = 1.75
    height = 2*width
    yint = 0.35
    xint = 0.2

    height2 = (rows/cols)*width

    ax01 = fi.make_axes([xmargin, ymargin+height+yint, width, height2])
    ax02 = fi.make_axes([xmargin+width+xint, ymargin+height+yint, width, height2])
    axcb = fi.make_axes([xmargin+2*width+2*xint, ymargin+height+yint+height2/2, width, 0.1*width])

    ax11 = fi.make_axes([xmargin, ymargin, width, height])
    ax21 = fi.make_axes([xmargin+width+xint, ymargin, width, height])
    ax31 = fi.make_axes([xmargin+2*width+2*xint, ymargin, width, height])

    cmap = plt.get_cmap('PuOr_r')
    cnorm  = colors.Normalize(vmin=-15, vmax=15)
    scalarMap = cm.ScalarMappable(norm=cnorm, cmap=cmap)

    mpl.colorbar.ColorbarBase(axcb, cmap=cmap, norm=cnorm, orientation='horizontal')
    axcb.set_xlabel(r'photocurrent (nA)')

    data = [Vg, Vsd, d, r]

    '''
    Selection Area
    '''
    points = [(32, 35) ,(49, 31), (51, 49)]
    cls = ['C0', 'C1', 'C3']
    disp_pc_ref(ax01, ax02, data, points, cls)

    '''
    Phase Map Portion
    '''
    phase_map_cuts(ax11, data, points[0], cls[0], cmap, cnorm, r"WSe$_{2}$ Contact", showy=True)
    phase_map_cuts(ax21, data, points[1], cls[1], cmap, cnorm, "Heterostructure")
    phase_map_cuts(ax31, data, points[2], cls[2], cmap, cnorm, r"MoSe$_{2}$ Contact")

    x1 = 0.07
    x2 = 0.39
    x3 = 0.69
    y1 = 0.95
    y2 = 0.66
    lblparams = {'fontsize':16, 'weight':'bold'}
    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x1, y2, 'b', **lblparams)
    plt.figtext(x2, y2, 'c', **lblparams)
    plt.figtext(x3, y2, 'd', **lblparams)

    if save:
        plt.savefig(join(svfile, 'figS46.png'), dpi=300)

    plt.show()
