'''
figS48.py

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
A visualization script to display supplementary Fig. S4.8 from the paper 'Stacking
enabled strong coupling of atomic motion to interlayer excitons in van der Waals
heterojunction photodiodes'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from os.path import join

from imaging_datasets import load_image_dataset

import qmodisplay as display

def plot_phase_map(ax, dataset, cmap, cnorm, p):
    Vsd, Vg, d, pc = dataset
    rows, cols, Mg, Mb = d.shape

    ax.imshow(pc, extent=[np.min(Vg), np.max(Vg), np.max(Vsd), np.min(Vsd)], cmap=cmap, norm=cnorm, interpolation='bilinear', aspect='auto')
    ax.plot(Vg, np.polyval(p,Vg), lw=1, color='r')

    ax.set_ylim(np.min(Vsd), np.max(Vsd))
    ax.tick_params(axis='y', pad=0.5)
    ax.set_yticks([0, -0.2, -0.4, -0.6])
    ax.set_ylabel(r'$V_{\mathrm{SD}}$ (V)', labelpad=-5)

    ax.set_xlim(np.min(Vg), np.max(Vg))
    ax.set_xticks([-1, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(['-1', '-0.5', '0', '0.5', '1'])
    ax.set_xlabel(r'$V_{\mathrm{G}}$ (V)')
# end plot_phase_map

def show_line_cuts(ax, dataset, Vcuts):
    Vsd, Vg, d, pc = dataset
    rows, cols, Mg, Mb = d.shape
    Vg_ = np.linspace(np.min(Vg), np.max(Vg), 200)
    cmap, cnorm, smap = display.colorscale_map(Vcuts, 'viridis')
    txtprops = {'va':'center', 'ha':'left', 'transform':ax.transAxes}
    ax.text(0.02, 0.92, r"$V_{\mathrm{SD}}=$", **txtprops)
    for i in range(len(Vcuts)):
        ix = np.searchsorted(Vsd, Vcuts[i])
        cval = smap.to_rgba(Vsd[ix])
        ax.plot(Vg, pc[ix,:], color=cval, lw=0.75)

        Ifunc = interp1d(Vg, pc[ix,:], kind='cubic')
        ixmax = np.argmax(Ifunc(Vg_))
        Vgmax = Vg_[ixmax]
        ax.plot(Vgmax, Ifunc(Vgmax), 'o', color='darkgreen', mec='k', ms=3)

        ax.text(0.02, 0.80-i*0.11, str(round(Vcuts[i],1))+' V', color=cval, **txtprops)
    ax.set_xlim(np.min(Vg), np.max(Vg))
    ax.set_ylim(0,13)
    ax.set_yticks([0,5,10])
    ax.set_yticklabels(["0   ", "5   ", "10  "])
    ax.set_xlabel(r"$V_{\mathrm{G}}$ (V)")
    ax.set_ylabel(r"$I_{\mathrm{PC}}$ (nA)", labelpad=1)
# end show_line_cuts

def show_Vgmax(ax, dataset, V1=-0.1):
    Vsd, Vg, d, pc = dataset
    rows, cols = pc.shape
    Vg_ = np.linspace(np.min(Vg), np.max(Vg),200)
    ix1 = np.searchsorted(Vsd, V1)
    Vgmax = np.zeros(rows)
    for i in range(ix1):
        Ifunc = interp1d(Vg, pc[i,:], kind='cubic')
        ixmax = np.argmax(Ifunc(Vg_))
        Vgmax[i] = Vg_[ixmax]
        ax.plot(Vgmax[i], Vsd[i], 'o', color='darkgreen', mec='k', ms=3)
    p = np.polyfit(Vgmax[:ix1], Vsd[:ix1], 1)
    ax.plot(Vgmax, np.polyval(p,Vgmax), lw=1, color='r')
    ax.set_ylabel(r'$V_{\mathrm{SD}}$ (V)', labelpad=1)
    ax.set_xlabel(r'$V_{\mathrm{G}}^{*}$ (V)', labelpad=0)
    ax.set_ylim(-0.8, -0.05)
    return p
# end find_Vg_of_max

if __name__ == '__main__':
    save, svfile = display.argsave()

    dataset = load_image_dataset()

    display.paper_figure_format(fntsize=10, labelpad=3)

    xinches = 4.3
    yinches = 3.7
    fi = display.figure_inches('figS48', xinches, yinches)
    xmargin = 0.7
    ymargin = 0.5

    xint = 0.65
    yint = 0.5

    width = 3.0

    width2 = 1.25
    yint2 = width - 2*width2

    ystart = ymargin

    xint3 = 0.1
    width3 = (width - xint3)/2
    ax1 = fi.make_axes([xmargin, ystart, width3, width])

    cbwidth = 0.05*width3
    cbheight = 0.25*width
    cbmargin = 0.05*width3
    axcb = fi.make_axes([xmargin+cbmargin, ystart+width-cbmargin-cbheight, cbwidth, cbheight], zorder=10)

    ax31 = fi.make_axes([xmargin+width3+xint, ystart+width2+yint2, width2, width2])
    ax32 = fi.make_axes([xmargin+width3+xint, ystart, width2, width2])

    cmap, cnorm, smap = display.colorscale_map(dataset[3], mapname='PuOr', cmin=-10, cmax=10)
    cb = display.make_colorbar(axcb, cmap, cnorm, ticks=[-10, -5, 0, 5, 10])
    axcb.set_ylabel(r'$I_{\mathrm{PC}}$ (nA)', labelpad=0)

    show_line_cuts(ax31, dataset, np.arange(-0.2, -0.71, -0.1))
    [m,b] = show_Vgmax(ax32, dataset)

    plot_phase_map(ax1, dataset, cmap, cnorm, [m,b])


    lblparams = {'fontsize':16, 'weight':'bold'}
    y1 = 0.94
    y2 = 0.45
    plt.figtext(0.08, y1, 'a', **lblparams)
    plt.figtext(0.6, y1, 'b', **lblparams)
    plt.figtext(0.6, y2, 'c', **lblparams)

    if save:
        plt.savefig(join(svfile, 'figS48.png'), dpi=300)

    plt.show()
