'''
fig2.py

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
A visualization script to display Fig. 2 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

from numpy.fft import fft2, fftshift, fftfreq

import qmodisplay as display

from qmomath import dydx, lorentzian
from qmomath import generic_fit

from os.path import join

def load_lowE_data(pcmin=82.6):
    dir = join('datasets','low-temp')
    Vg = np.loadtxt(join(dir, 'indirect-exciton-Vg.txt'))
    Eph = np.loadtxt(join(dir, 'indirect-exciton-Eph.txt'))
    pc = np.loadtxt(join(dir, 'indirect-exciton-pc.txt'))
    pc = 1.0e2*pc # convert to pA
    pc = pc - pcmin

    rows, cols = pc.shape
    dpc = np.zeros(pc.shape)
    d2pc = np.zeros(pc.shape)
    for j in range(cols):
        dpc[:,j] = dydx(Eph, pc[:,j])
        d2pc[:,j] = dydx(Eph, dpc[:,j])
    #
    return Eph, Vg, pc, dpc, d2pc
# load_data

def load_highE_data(pcmin=50):
    dir = join('datasets','low-temp')
    Vg = np.loadtxt(join(dir, 'direct-exciton-Vg.txt'))
    Eph = np.loadtxt(join(dir, 'direct-exciton-Eph.txt'))
    pc = np.loadtxt(join(dir, 'direct-exciton-pc.txt'))
    pc = 1.0e2*pc # convert to pA
    pc = pc - pcmin
    ix1 = np.searchsorted(Vg, -16)
    ix2 = np.searchsorted(Vg, 16)
    Vg = Vg[ix1:ix2]
    pc = pc[:,ix1:ix2]
    rows, cols = pc.shape

    ixc = rows -  np.searchsorted(Eph[::-1], 1.4)
    Eph = Eph[ixc:]
    pc = pc[ixc:,:]

    rows, cols = pc.shape
    dpc = np.zeros(pc.shape)
    d2pc = np.zeros(pc.shape)
    for j in range(cols):
        dpc[:,j] = dydx(Eph, pc[:,j])
        d2pc[:,j] = dydx(Eph, dpc[:,j])
    return Eph, Vg, pc, dpc, d2pc
# load_data

def show_pc_map(ax, axcb, dataset, c='w', vmax=1.65):
    Eph, Vg, pc, dpc, d2pc = dataset
    pcmap, pcnorm, pcsmap = display.colorscale_map(pc, cmin=0.0, cmax=vmax, mapname='inferno')
    rcParams['axes.edgecolor'] = c
    display.make_colorbar(axcb, pcmap, pcnorm, orientation='vertical', ticks=np.arange(0,vmax+0.1,0.5))
    axcb.yaxis.tick_left()
    axcb.yaxis.set_label_position('left')
    display.change_axes_colors(axcb, c)
    axcb.set_ylabel('photocurrent $I_{\mathrm{PC}}$ (pA)', labelpad=1)

    ax.imshow(pc, cmap=pcmap, norm=pcnorm, extent=[np.min(Vg), np.max(Vg), np.min(Eph), np.max(Eph)], aspect='auto', interpolation='none')
    ax.set_xlim(-15,15)
    ax.set_ylabel(r'$E_{\mathrm{PH}}$ (eV)')
    ax.set_xlabel(r'$V_{\mathrm{G}}$ (V)')
# end show_pc_map

def show_lowE_line_cuts(ax, dataset, V1=-2, V2=3, y1=0.15, y2=1.6):
    Eph, Vg, pc, dpc, d2pc = dataset
    ix1 = np.searchsorted(Vg, V1)
    ix2 = np.searchsorted(Vg, V2)
    Vmap, Vnorm, Vsmap = display.colorscale_map(Vg[ix1-2:ix2+2], mapname='viridis', truncate=(0,0.8))
    txtprops = {'va':'center', 'ha':'left', 'transform':ax.transAxes}
    ax.text(0.04, 0.965, r"$V_{\mathrm{G}}=$", **txtprops)
    for j in range(ix1,ix2+1):
        cval = Vsmap.to_rgba(Vg[j])
        ax.plot(Eph, pc[:,j], '.-', color=cval)
        lbl = str(round(Vg[j],1))+' V'
        if Vg[j] > 0:
            lbl = ' ' + lbl
        ax.text(0.04, 0.91-(j-ix1)*0.05, lbl, color=cval, **txtprops)
        #axmap.axvline(Vg[j], color=cval)
    ax.set_xticks([])
    ax.set_xticks(np.arange(0.9, 1.31, 0.05))
    ax.set_xlim(np.min(Eph), np.max(Eph))
    ax.set_ylim(0,1.6)
    ax.set_yticks(np.arange(0.0, 3.0, 0.2))
    ax.set_ylim(y1, y2)
    ax.set_ylabel(r'photocurrent $I_{\mathrm{PC}}$ (pA)')
    ax.set_xlabel(r'$E_{\mathrm{PH}}$ (eV)')
# end show_line_cuts

def show_highE_line_cuts(ax, dataset, V1=-4, V2=2, y1=1.15, y2=3.0):
    Eph, Vg, pc, dpc, d2pc = dataset
    ix1 = np.searchsorted(Vg, V1)
    ix2 = np.searchsorted(Vg, V2)
    Vmap, Vnorm, Vsmap = display.colorscale_map(Vg[ix1-2:ix2+2], mapname='viridis', truncate=(0,0.8))
    txtprops = {'va':'center', 'ha':'left', 'transform':ax.transAxes}
    ax.text(0.04, 0.965, r"$V_{\mathrm{G}}=$", **txtprops)
    for j in range(ix1,ix2+1):
        cval = Vsmap.to_rgba(Vg[j])
        ax.plot(Eph, pc[:,j], '.-', color=cval)
        lbl = str(round(Vg[j],1))+' V'
        if Vg[j] > 0:
            lbl = ' ' + lbl
        if j < ix1 + 4:
            ax.text(0.04, 0.91-(j-ix1)*0.05, lbl, color=cval, **txtprops)
        else:
            ax.text(0.35, 0.91-(j-ix1-4)*0.05, lbl, color=cval, **txtprops)
        #axmap.axvline(Vg[j], color=cval)
    ax.set_xticks([])
    ax.set_xticks(np.arange(1.24, 1.41, 0.05))
    ax.set_xlim(np.min(Eph), np.max(Eph))
    ax.set_yticks(np.arange(0.6, 4.0, 0.2))
    ax.set_ylim(y1, y2)
    ax.set_ylabel(r'photocurrent $I_{\mathrm{PC}}$ (pA)')
    ax.set_xlabel(r'$E_{\mathrm{PH}}$ (eV)')
# end show_line_cuts

def guide_to_eye(ax, x0, y0, dy=0.1):
    dy2 = dy / 2
    ax.annotate('', xy=(x0, y0), xytext=(x0-0.030, y0), arrowprops=dict(arrowstyle="<|-|>", color='k'), annotation_clip=False)
    ax.annotate('', xy=(x0, y0-dy2), xytext=(x0, y0+dy2), arrowprops=dict(arrowstyle="-", color='k'), annotation_clip=False)
    ax.annotate('', xy=(x0-0.030, y0-dy2), xytext=(x0-0.030, y0+dy2), arrowprops=dict(arrowstyle="-", color='k'), annotation_clip=False)
    ax.annotate('30'+'\n'+'meV', xy=(x0-0.015, y0+dy/2), annotation_clip=False, ha='center', color='k')

def show_highE_fit_cut(ax, dataset, V1=-4, Estart=1.253):
    def sum_lorentz(x, *p):
        ret = None
        for i in range(0, 5):
            ix = 4*i
            if ret is None:
                ret = lorentzian(x, p[ix], p[ix+1], p[ix+2], p[ix+3])
            else:
                ret = ret + lorentzian(x, p[ix], p[ix+1], p[ix+2], p[ix+3])
        # Need to introduce a penalty to deal with overfitting making parameters negative
        if any(param < 0 for param in p):
            penalty = 100000000000000.0
        elif p[ix] < 1e-4:
            penalty = 100000000000000.0
        else:
            penalty = 1.0
        return penalty*ret

    Eph, Vg, pc, dpc, d2pc = dataset
    ix1 = np.searchsorted(Vg, V1)
    Vmap, Vnorm, Vsmap = display.colorscale_map(Vg[ix1-2:ix1+5], mapname='viridis', truncate=(0,0.8))
    txtprops = {'va':'center', 'ha':'left', 'transform':ax.transAxes}
    ax.text(0.04, 0.95, r"$V_{\mathrm{G}}=$ " + str(round(Vg[ix1],1)) + " V", **txtprops)
    ax.plot(Eph, pc[:,ix1], 'o', color=Vsmap.to_rgba(V1))

    p0 = [0.01, Estart, 0.01, 1.2]
    for i in range(1, 5):
        p0.extend([0.01, Estart+i*0.030, 0.01, 0])
    p, perr = generic_fit(Eph, pc[:,ix1], p0, sum_lorentz)

    xx = np.linspace(np.min(Eph), np.max(Eph), 200)
    ft = sum_lorentz(xx, *p)
    ax.plot(xx, ft, 'k')
    for i in range(5):
        ix = 4*i
        #print(p[ix:ix+3])
        ft = lorentzian(xx, p[ix], p[ix+1], p[ix+2], 1.6)
        ax.plot(xx, ft, '-', color='lightgray')
    ax.set_xticks([])
    ax.set_xticks(np.arange(1.24, 1.41, 0.05))
    ax.set_xlim(np.min(Eph), np.max(Eph))
    ax.set_yticks(np.arange(0.6, 4.0, 0.2))
    ax.set_ylim(1.6, 2.8)
    ax.set_ylabel(r'photocurrent $I_{\mathrm{PC}}$ (pA)')
    ax.set_xlabel(r'$E_{\mathrm{PH}}$ (eV)')
    guide_to_eye(ax, 1.379, 2.54)
# end show_line_cuts

def show_lowE_fit_cut(ax, dataset, V1=-2, Estart=0.89):
    def sum_lorentz(x, *p):
        ret = None
        for i in range(0, 5):
            ix = 4*i
            if ret is None:
                ret = lorentzian(x, np.abs(p[ix]), p[ix+1], p[ix+2], p[ix+3])
            else:
                ret = ret + lorentzian(x, np.abs(p[ix]), p[ix+1], p[ix+2], p[ix+3])
            penalty = 1.0
        return penalty*ret

    Eph, Vg, pc, dpc, d2pc = dataset
    ix1 = np.searchsorted(Vg, V1)
    Vmap, Vnorm, Vsmap = display.colorscale_map(Vg[ix1-2:ix1+5], mapname='viridis', truncate=(0,0.8))
    txtprops = {'va':'center', 'ha':'left', 'transform':ax.transAxes}
    ax.text(0.04, 0.95, r"$V_{\mathrm{G}}=$ " + str(round(Vg[ix1],1)) + " V", **txtprops)
    ax.plot(Eph, pc[:,ix1], 'o', color=Vsmap.to_rgba(V1))

    p0 = [0.01, Estart, 0.01, 1.2]
    for i in range(1, 5):
        p0.extend([0.01, Estart+i*0.030, 0.01, 0])
    p, perr = generic_fit(Eph, pc[:,ix1], p0, sum_lorentz)

    xx = np.linspace(np.min(Eph), np.max(Eph), 200)
    ft = sum_lorentz(xx, *p)
    ax.plot(xx, ft, 'k')
    for i in range(5):
        ix = 4*i
        #print(p[ix:ix+3])
        ft = lorentzian(xx, np.abs(p[ix]), p[ix+1], p[ix+2], 0.2)
        ax.plot(xx, ft, '-', color='lightgray')
    ax.set_xticks([])
    ax.set_xticks(np.arange(0.9, 1.31, 0.05))
    ax.set_xlim(np.min(Eph), np.max(Eph))
    ax.set_ylim(0.2,1.6)
    # ax.set_yticks(np.arange(0.0, 1.0, 0.2))
    ax.set_ylabel(r'photocurrent $I_{\mathrm{PC}}$ (pA)')
    ax.set_xlabel(r'$E_{\mathrm{PH}}$ (eV)')
    guide_to_eye(ax, 0.922, 0.73)

# end show_line_cuts

def show_fft(ax, axcb, dataset, c='w', show22=False):
    Eph, Vg, pc, dpc, d2pc = dataset
    rows, cols = d2pc.shape
    d2pc_fft = np.abs(fftshift(fft2(d2pc)))
    d2pc_fft = 1.111*d2pc_fft/np.max(d2pc_fft)

    frows = fftfreq(rows, d=np.mean(np.diff(Eph)))
    fcols = fftfreq(cols, d=np.mean(np.diff(Vg)))

    fcmap, fcnorm, fcsmap = display.colorscale_map(d2pc_fft, cmin=0, cmax=1, mapname='inferno')
    rcParams['axes.edgecolor'] = c
    display.make_colorbar(axcb, fcmap, fcnorm, orientation='vertical')
    display.change_axes_colors(axcb, c)
    axcb.set_ylabel(r'd$^{2} I_{\mathrm{PC}}$/d$E^{2}_{\mathrm{PH}}$ FT (arb.)')

    exnt = [np.min(fcols), np.max(fcols), np.min(frows), np.max(frows)]
    ax.imshow(d2pc_fft, cmap=fcmap, norm=fcnorm, extent=exnt, aspect='auto', interpolation='none')
    ax.set_xlabel(r'$V_{\mathrm{G}}$ Fourier component (V$^{-1}$)')
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.32, 0.32)
    ax.axhline(1/0.030, color='C0', ls='--')
    ax.text(-0.32, 28, r"(30 meV)$^{-1}$", ha='left', color='C0', fontsize=10)
    if show22:
        ax.axhline(1/0.022, color='C2', ls='--')
        ax.text(-0.32, 48, r"(22 meV)$^{-1}$", ha='left', color='C2', fontsize=10)
    ax.set_ylabel(r'$E_{\mathrm{PH}}$ Fourier component (eV$^{-1}$)', labelpad=-2)
# show_fft

def show_center_schematic(ax):
    image1 = plt.imread(join('..','schematics','interlayer exciton 2.png'))
    ax.axis('off')
    ax.imshow(image1, aspect='auto')
    txtparams = {'ha':'left', 'va':'center'}
    ax.text(85, 160, r"WSe$_{2}$", color='k', **txtparams)
    ax.text(85, 350, r"WSe$_{2}$", color='k', **txtparams)
    ax.text(1500, 545, r"MoSe$_{2}$", color='k', **txtparams)

    txtparams = {'ha':'center', 'va':'center'}
    ax.text(2070, 600, r"K, WSe$_{2}$", color='k', **txtparams)
    ax.text(2620, 600, r"$\Gamma$, WSe$_{2}$", color='k', **txtparams)
    ax.text(2070, 150, r"K, MoSe$_{2}$", color='k', **txtparams)

    ax.text(2465, 295, r"$\Gamma \rightarrow K$", color='#ff0000', **txtparams)
    ax.text(1900, 390, r"$K \rightarrow K$", color='#0000ff', **txtparams)

    arr = FancyArrowPatch((1200,520), (1200,200), arrowstyle="-|>", color='k', lw=2, mutation_scale=10)
    ax.text(1270, 360, r"$\mathbf{p}$", color='k', fontsize=12, **txtparams)
    ax.add_patch(arr)
# end show_band_schematic

if __name__ == '__main__':
    save, svfile = display.argsave()

    dataset_top = load_highE_data()
    dataset_bot = load_lowE_data()

    display.paper_figure_format(fntsize=10, labelpad=3)

    xinches = 6.5
    yinches = 9.0
    fi = display.figure_inches('fig2', xinches, yinches)
    fig1 = fi.get_fig()
    xmargin = 0.6
    ymargin = 0.45

    height = 3.0 #3.5
    width = height/2
    xint = 0.58

    cbwidth = 0.07*width
    cbheight = 0.33*height
    cbmargin = 0.04*width

    ystart = yinches - ymargin/2.5 - height
    ax1 = fi.make_axes([xmargin, ystart, width, height])
    ax1cb = fi.make_axes([xmargin+width-cbmargin-cbwidth, ystart+4*cbmargin, cbwidth, cbheight])
    ax2 = fi.make_axes([xmargin+xint+width, ystart, width, height])
    ax3 = fi.make_axes([xmargin+2*xint+2*width, ystart, width, height])
    ax3cb = fi.make_axes([xmargin+2*xint+2*width+cbmargin, ystart+height-cbheight-3*cbmargin, cbwidth, cbheight])

    centerY = (ystart + ymargin + height)/2 - 0.15

    widthc = 2*xint+3*width
    heightc = (717/2837)*widthc
    ax4 = fi.make_axes([xmargin, centerY-heightc/2, widthc, heightc])

    ax6 = fi.make_axes([xmargin, ymargin, width, height])
    ax6cb = fi.make_axes([xmargin+width-cbmargin-cbwidth, ymargin+4*cbmargin, cbwidth, cbheight])
    ax7 = fi.make_axes([xmargin+xint+width, ymargin, width, height])
    ax8 = fi.make_axes([xmargin+2*xint+2*width, ymargin, width, height])
    ax8cb = fi.make_axes([xmargin+2*xint+2*width+cbmargin, ymargin+height-cbheight-3*cbmargin, cbwidth, cbheight])

    show_pc_map(ax1, ax1cb, dataset_top, vmax=2.75)
    show_highE_fit_cut(ax2, dataset_top)
    show_fft(ax3, ax3cb, dataset_top)

    show_center_schematic(ax4)

    show_pc_map(ax6, ax6cb, dataset_bot)
    show_lowE_fit_cut(ax7, dataset_bot)
    show_fft(ax8, ax8cb, dataset_bot, show22=True)

    ax1.set_title(r"$K \rightarrow K$", fontsize=12)
    ax6.set_title(r"$\Gamma \rightarrow K$", fontsize=12)

    lblparams = {'fontsize':16, 'weight':'bold'}
    x1 = 0.04
    x2 = 0.339
    x3 = 0.654
    y1 = 0.98
    y2 = 0.58
    y3 = 0.38
    plt.figtext(x1, y1, 'a', **lblparams)
    plt.figtext(x2, y1, 'b', **lblparams)
    plt.figtext(x3, y1, 'c', **lblparams)

    plt.figtext(x1, y2, 'd', **lblparams)

    plt.figtext(x1, y3, 'e', **lblparams)
    plt.figtext(x2, y3, 'f', **lblparams)
    plt.figtext(x3, y3, 'g', **lblparams)

    if save:
        plt.savefig(join(svfile, 'fig2.png'), dpi=300)

    plt.show()
