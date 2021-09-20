'''
fig1.py

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
A visualization script to display Fig. 1 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np
import matplotlib.pyplot as plt

from qmomath import gauss, generic_fit

import matplotlib.patheffects as path_effects
from matplotlib import cm
from matplotlib.colors import ListedColormap

from os.path import join

from imaging_datasets import load_image_dataset
import qmodisplay as display

# c = [x1, x2, y1, y2]
def avg_region(c, data):
    return np.mean(data[c[2]:c[3],c[0]:c[1]])
#

def plot_Raman_data(ax, axin, inx1=25.75, inx2=35.25, centerb=30.0, x1=20, x2=75, y1=-1, y2=70.1, N=200, sigma=4):
    fl = join("..",'datasets', 'Raman-PL data')
    MoSe2 = np.loadtxt(join(fl, 'MoSe2_Raman.txt'))
    WSe2 = np.loadtxt(join(fl, 'WSe2_Raman.txt'))
    HS = np.loadtxt(join(fl, 'HS_Raman.txt'))

    data = [MoSe2, WSe2, HS]
    colors = ['#27a47c', '#0800cf', 'k']
    for i in range(len(data)):
        data[i][:,0] = 1e3*data[i][::-1,0]
        data[i][:,1] = 1e-2*data[i][::-1,1]
        ax.plot(data[i][:,0], data[i][:,1], color=colors[i], lw=1)
        ix1 = np.searchsorted(data[i][:,0], inx1)
        ix2 = np.searchsorted(data[i][:,0], inx2)
        axin.semilogy(data[i][ix1:ix2,0], data[i][ix1:ix2,1], color=colors[i], lw=1)
    ax.set_xlim(x1,x2)
    ax.set_ylim(y1, y2)
    ax.set_xticks(np.arange(20,80,10))
    ytks = np.arange(20,80,20)
    ax.set_yticks(ytks)
    ytklbls = []
    for i in range(len(ytks)):
        ytklbls.append('  ' + str(ytks[i]))
    ax.set_yticklabels(ytklbls)
    ax.set_xlabel('Raman shift (meV)')
    ax.set_ylabel('Raman intensity (arb.)', labelpad=-1)

    V = np.zeros((N, N))
    x = np.linspace(x1, x2, N)
    for i in range(N):
        V[i,:] = gauss(x, 1.0, centerb, sigma)
    #ax.imshow(np.flipud(V), cmap=getcmap(), vmin=0, vmax=2.0, extent=[x1, x2, y1, y2], aspect='auto')
    ax.imshow(np.flipud(V), cmap=getcmap(), vmin=0, vmax=2.7, extent=[x1, x2, y1, y2], aspect='auto')

    axin.set_xlim(inx1,inx2)
    axin.set_ylim(0.36, 74)
    axin.set_xticks([26,30,34])
    axin.set_xlabel('shift (meV)', labelpad=2)
    axin.set_ylabel('intensity (arb.)', labelpad=1)
# end plot_PL_data

def plot_PL_data(ax):
    fl = join("..",'datasets', 'Raman-PL data')
    MoSe2 = np.loadtxt(join(fl, 'MoSe2_PL.txt'))
    WSe2 = np.loadtxt(join(fl, 'WSe2_PL.txt'))
    HS = np.loadtxt(join(fl, 'HS_PL.txt'))

    data = [MoSe2, WSe2, HS]
    colors = ['#27a47c', '#0800cf', 'k']
    for i in range(len(data)):
        data[i][:,1] = 1e-2*data[i][:,1]
        ax.semilogy(data[i][:,0], data[i][:,1], color=colors[i], lw=1)
    ax.set_xlim(1.2, 2.0)
    ax.set_ylim(0.37, 20)
    ax.set_xlabel('photon energy (eV)')
    ax.set_ylabel('photoluminescence (arb.)')

    txtparams = {'weight':'bold', 'ha':'center', 'va':'center', 'fontsize':11}
    ax.text(1.4, 7.0, r"WSe$_{2}$", color='#0800cf', **txtparams)
    ax.text(1.65, 0.5, r"heterostructure", color='k', **txtparams)
    ax.text(1.85, 4.2, r"MoSe$_{2}$", color='#27a47c', **txtparams)
# end plot_PL_data

def show_schematics(ax1, ax2):
    txtparams = {'ha':'center', 'va':'center', 'weight':'bold', 'fontsize':11}
    image1 = plt.imread(join('..','schematics','sample-diagram.png'))
    ax1.axis('off')
    ax1.imshow(image1, aspect='auto')
    ax1.text(183, 69, r"$V_{\mathrm{SD}}$", **txtparams)
    ax1.text(1052, 69, r"$I$", **txtparams)
    ax1.text(994, 812, r"$V_{\mathrm{G}}$", **txtparams)
    ax1.text(1074, 493, r"WSe$_2$", **txtparams)
    ax1.text(168, 493, r"MoSe$_2$", **txtparams)
    ax1.text(168, 237, r"hBN", **txtparams)
    ax1.text(168, 770, r"graphene", **txtparams)

    image2 = plt.imread(join('..','schematics','band-diagram.png'))
    ax2.axis('off')
    ax2.imshow(image2, aspect='auto')
    ax2.text(994, 785, r"2L-WSe$_2$", **txtparams)
    ax2.text(223, 785, r"MoSe$_2$", **txtparams)
    ax2.text(1130, 409, r"$\mu_{c}$", color='grey', **txtparams)
    ax2.text(675, 353, r"$E_{\mathrm{I}}$", **txtparams)
    # ax2.text(490, 370, r"$E_{\mathrm{PH}}$", color='#7a3a3a', **txtparams)

    ax2.text(53, 272+40, r"CB", ha='left', va='center', weight='bold')
    ax2.text(53, 715-30, r"VB", ha='left', va='center', weight='bold')
    ax2.text(1183, 170+40, r"CB", ha='right', va='center', weight='bold')
    ax2.text(1183, 562-30, r"VB", ha='right', va='center', weight='bold')

    txtparams['fontsize'] = 14
    ax2.text(17, 30, r"$\epsilon$", **txtparams)
    ax2.text(223, 55, r"n", color='w', path_effects=[path_effects.withSimplePatchShadow()], **txtparams)
    ax2.text(994, 55, r"p", color='w', path_effects=[path_effects.withSimplePatchShadow()], **txtparams)
# end show_schematics

def show_device_schematic(ax):
    txtparams = {'ha':'center', 'va':'center', 'weight':'bold', 'fontsize':10}
    image1 = plt.imread(join('..','schematics','Fig1A schematic.png'))
    ax.axis('off')
    ax.imshow(image1, aspect='auto')
    ax.text(160, 515, r"Graphite", **txtparams)
    ax.text(160, 295, r"WSe$_2$ Contact", **txtparams)
    ax.text(495, 135, r"WSe$_2$", **txtparams)
    ax.text(900, 193, r"MoSe$_2$", **txtparams)
    ax.text(1100, 130, r"Top h-BN", **txtparams)
    ax.text(1100, 300, r"MoSe$_2$ Contact", **txtparams)
    ax.text(1150, 460, r"Bottom h-BN", **txtparams)
    ax.text(1100, 563, r"Gate Contacts", **txtparams)
#

def getcmap():
    blues = cm.get_cmap('Blues', 100)
    cvals = np.ones((101,4))
    cvals[1:,0] = blues(np.linspace(0, 1, 100))[:,0]
    cvals[1:,1] = blues(np.linspace(0, 1, 100))[:,1]
    cvals[1:,2] = blues(np.linspace(0, 1, 100))[:,2]
    return ListedColormap(cvals)
# end getcmap

def diode(V, Is, n, kT=0.02559):
    return Is*(np.exp(V/(n*kT))-1)
# end diode

def show_dark_current(ax, rn='2019_05_16_10'):
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
    ax.set_ylabel(r"interlayer dark current (nA)")
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-30, 900)

    s = r"$\frac{I}{I_{0}} = exp \left( \frac{eV_{\mathrm{SD}}}{\alpha k_{B} T} \right) - 1$"
    s = s + ", " + r"$\alpha =$" + str(round(p[1],3))
    ax.text(-0.19, 125, s, color='r', va='center', ma='center', fontsize=11)

    return p[1]
# end show_dark_current

def plot_avg_pc(ax, Vg1=-0.3, Vg2=0.4):
    dataset = load_image_dataset()
    Vsd, Vg, d, phase = dataset
    ix1 = np.searchsorted(Vg, Vg1)
    ix2 = np.searchsorted(Vg, Vg2)
    pc = np.mean(phase[:,ix1:ix2], axis=1)

    ax.plot(Vsd, pc, color='C4')
    ax.set_xlim(np.min(Vsd), np.max(Vsd))
    ax.set_xlabel(r'$V_{\mathrm{SD}}$ (V)')
    ax.set_ylabel('photocurrent $I_{\mathrm{PC}}$ (nA)')
    display.yaxis_right(ax)
# end plot_avg_d2pc

if __name__ == '__main__':
    save, svfile = display.argsave()

    display.paper_figure_format(fntsize=10, labelpad=5)

    xinches = 10.75
    yinches = 5.8
    fi = display.figure_inches('fig1', xinches, yinches)

    xmargin = 0.65
    ymargin = 0.60

    xint = 0.65
    yint = 0.33
    width = 2.75
    # width2 = 3.5
    height2 = (2.5/3.5)*width

    ax1 = fi.make_axes([xmargin, ymargin+width+yint, width, height2])
    ax2 = fi.make_axes([xmargin+xint+width, ymargin+width+yint, width, height2])
    ax3 = fi.make_axes([xmargin+2*xint+2*width, ymargin+width+yint, width, height2])

    ax4 = fi.make_axes([xmargin, ymargin, width, width])
    widthin = 0.5*width
    margin = 0.03*width
    ax4in = fi.make_axes([xmargin+width-widthin-margin, ymargin+width-widthin-margin, widthin, widthin])

    ax5 = fi.make_axes([xmargin+xint+width, ymargin, width, width])
    ax6 = fi.make_axes([xmargin+2*xint+2*width, ymargin, width, width])
    ax6in = fi.make_axes([xmargin+2*width+2*xint+margin, ymargin+width-widthin-margin, widthin, widthin])

    show_device_schematic(ax1)
    show_schematics(ax2, ax3)

    plot_Raman_data(ax4, ax4in)
    plot_PL_data(ax5)

    alpha = show_dark_current(ax6)
    plot_avg_pc(ax6in)

    lblparams = {'fontsize':16, 'weight':'bold'}
    x1 = 0.03
    x2 = 0.35
    x3 = 0.67
    y1 = 0.96
    y2 = 0.56

    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x2, y1, 'b', **lblparams)
    plt.figtext(x3, y1, 'c', **lblparams)
    plt.figtext(x1, y2, 'd', **lblparams)
    plt.figtext(x2, y2, 'e', **lblparams)
    plt.figtext(x3, y2, 'f', **lblparams)

    if save:
        plt.savefig(join(svfile, 'fig1.png'), dpi=300)

    plt.show()
