'''
figS44.py

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
A visualization script to display supplementary Fig. S4.4 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.path import Path
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import inset_locator

from os.path import join

import qmodisplay as display
from imaging_datasets import load_image_dataset

def tile_plot(ax1, dataset, cmap, cnorm, spinewidth=0.2):
    Vsd, Vg, d = dataset
    rows, cols, Mg, Mb = d.shape

    rcoords = np.array(range(0,Mg,1))
    ccoords = np.array(range(0,Mb,1))

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
            ax.imshow(d[:,8:92,i,j], cmap=cmap, norm=cnorm, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            cix += 1
        rix += 1
    #

    ax1.set_xlim(np.min(Vsd), np.max(Vsd))
    ax1.set_ylim(np.min(Vg), np.max(Vg))

    ax1.set_xticks(np.arange(-0.7,0.11,0.1))
    ax1.set_xlabel(r'V$_{SD}$ (V)')
    ax1.set_yticks([-1, -0.5, 0.0, 0.5, 1.0])
    ax1.set_ylabel(r'V$_{G}$ (V)', labelpad=1)
# end tile_plot

def plot_series(axes, dataset, Vsdvals, Vgval, cmap, cnorm, heteroarea, showxlabels=False, showVg=False, showscale=False, showhetero=True , ic1=25, ic2=75, ir1=15, ir2=65):
    Vsd, Vg, d = dataset
    rows, cols, Mg, Mb = d.shape
    N = len(Vsdvals)
    ixVg = np.searchsorted(Vg, Vgval)

    # Rough Scale
    # measured 3.06 pixels = 1 um
    micron = 3.06

    for i in range(N):
        ixVsd = np.searchsorted(Vsd, Vsdvals[i])
        ax = axes[i]
        #pc = np.rot90(d[:,:,ixVg,ixVsd])
        pc = d[:,:,ixVg,ixVsd]
        ax.imshow(pc, cmap=cmap, norm=cnorm, aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(ic1, ic2)
        ax.set_ylim(ir2, ir1)

        if showhetero:
            ax.add_patch(patches.PathPatch(heteroarea, ec='k', fill=False, lw=1))

        if showscale and i == N-1:
            ax.add_patch(patches.Circle((70, 60), 1.74*micron/2, facecolor='k')) # for 1250 nm, i.e. 1.67 micron beamspot times (1250/1200)
            ax.add_patch(patches.Rectangle((28, 60), 5*micron, 0.75*micron, facecolor='k'))
            ax.text(28+0.5*5*micron, 58, r'5 $\mu$m', ha='center')

        if i == 0:
            tstr = 'V$_{SD}$ = '
        else:
            tstr = ''
        tstr = tstr + str(round(Vsd[ixVsd],1)) + ' V'
        if showxlabels:
            ax.set_title(tstr, fontsize=14)
        #
        if showVg:
            axes[0].set_ylabel('V$_G$ = ' + str(round(Vg[ixVg],1)) + ' V', fontsize=14)
        else:
            axes[0].set_ylabel(str(round(Vg[ixVg],1)) + ' V', fontsize=14)
# end plot_series

if __name__ == '__main__':
    save, svfile = display.argsave()

    Vsd, Vg, d, hs = load_image_dataset("Vg_Vsd_narrow")
    rows, cols, Mg, Mb = d.shape
    N = Mg*Mb

    display.paper_figure_format(labelpad=3)

    yinches = 7.25
    xinches = 6.5
    fi = display.figure_inches('figS44', xinches, yinches)
    fig1 = fi.get_fig()
    xmargin = 0.9
    ymargin = 0.2
    xint = 0.25
    yint = 0.65

    ystart = 4.25
    width = 5.0
    height = (Mg/Mb)*width
    ax1 = fi.make_axes([xmargin, ystart, width, height])

    ax1cb = fi.make_axes([xmargin+0.7*width, ystart+0.15*height, 0.25*width, 0.05*height], zorder=200)

    width2 = 1.0
    yint2 = 0.1
    series = []
    for k in range(3):
        series.append([])
        for i in range(5):
            series[k].append(fi.make_axes([xmargin+i*width2, ymargin+k*(yint2+width2), width2, width2], zorder=12))
    #

    cmap, cnorm, smap = display.colorscale_map(d, mapname='PuOr_r', cmin=-10, cmax=10)
    display.make_colorbar(ax1cb, cmap, cnorm, orientation='horizontal', ticks=[-10, -5, 0, 5, 10])
    ax1cb.set_title('$I_{\mathrm{PC}}$ (nA)', ha='center')

    dataset = [Vsd, Vg, d]

    tile_plot(ax1, dataset, cmap, cnorm)

    # Heterostructure Region
    X = np.array([55, 52, 39, 41, 55])
    Y = np.array([34, 27, 28, 36, 34])

    verts = np.zeros((np.size(X),2))
    verts[:,0] = X
    verts[:,1] = Y
    heteroarea = Path(verts)

    Vsdvals = np.arange(-0.4, 0.01, 0.1)
    Vgvals = [0.8, 0.2, -0.4]

    plot_series(series[2], dataset, Vsdvals, Vgvals[0], cmap, cnorm, heteroarea, showxlabels=True)
    plot_series(series[1], dataset, Vsdvals, Vgvals[1], cmap, cnorm, heteroarea)
    plot_series(series[0], dataset, Vsdvals, Vgvals[2], cmap, cnorm, heteroarea, showscale=True, showVg=True)

    x1 = 0.05
    y1 = 0.97
    y2 = 0.47
    lblparams = {'fontsize':16, 'weight':'bold'}
    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x1, y2, 'b', **lblparams)

    if save:
        plt.savefig(join(svfile, 'figS44.png'), dpi=300)

    plt.show()
