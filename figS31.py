'''
figS31.py

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
A visualization script to display supplementary Fig. S3.1 from the paper 'Stacking enabled vibronic
exciton-phonon states in van der Waals heterojunctions'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np

from os.path import join
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

import qmodisplay as display
from qmomath import kb_eV, gauss

def getcmap():
    blues = cm.get_cmap('Blues', 100)
    greens = cm.get_cmap('Greens', 100)

    cvals = np.ones((201,4))

    cvals[0:100,0] = greens(np.linspace(1, 0, 100))[:,0]
    cvals[0:100,1] = greens(np.linspace(1, 0, 100))[:,1]
    cvals[0:100,2] = greens(np.linspace(1, 0, 100))[:,2]

    cvals[101:,0] = blues(np.linspace(0, 1, 100))[:,0]
    cvals[101:,1] = blues(np.linspace(0, 1, 100))[:,1]
    cvals[101:,2] = blues(np.linspace(0, 1, 100))[:,2]
    return ListedColormap(cvals)
# end getcmap

def load_dispersion(datafile='DFT-bands'):
    datafl = join('datasets','phonon-dft-calc',datafile)
    data = np.loadtxt(datafl+'.txt')
    ny, nx = data.shape
    bands = []
    ix0 = 0
    for i in range(1,ny):
        if data[i,0] == 0.0:
            bands.append(data[ix0:i-1,:])
            ix0 = i
    return bands
# end load_dispersion

def load_DOS(datafile):
    datafl = join('datasets','phonon-dft-calc',datafile)
    data = np.loadtxt(datafl+'.csv', delimiter=',')
    # Smooth out rough edges from extracted data
    srt = np.argsort(data[:,0])
    data = data[srt,:]
    data[:,1] = gaussian_filter(data[:,1], 2)
    return data
# end load_DOS

def thermal_scale(ax, centerb=30.0, centerg=22.0, T=20, e1=0, e2=40, N=500):
    V = np.zeros((N, N))
    y = np.linspace(e1, e2, N)
    sigma = np.sqrt(1000*kb_eV*T)
    x1, x2 = ax.get_xlim()
    for i in range(N):
        V[:,i] = gauss(y, 1.0, centerb, sigma)
        #V[:,i] = V[:,i] + gauss(y, -1.0, centerg, sigma)
    cmap = getcmap()
    ax.imshow(np.flipud(V), cmap=cmap, vmin=-1.65, vmax=1.65, extent=[x1, x2, e1, e2], aspect='auto')
# end thermal_scale

def plot_phonon_dispersion(ax1, ax2, bands, DOS):
    WSe2_DOS, MoSe2_DOS, Hetero_DOS = DOS
    N = len(bands)

    for i in range(N):
        band = bands[i]
        ax1.plot(band[:,0], band[:,1], 'k', lw=1.0)
    #

    ax1.set_xticks([0.0, 0.17625, 0.27445, 0.47])
    ax1.set_xticklabels([r'$\Gamma$','M','K', r'$\Gamma$'])
    ax1.set_xlim(0, 0.47)
    ax1.set_xlabel('wave vector', labelpad=2)
    ax1.set_ylabel('phonon energy (meV)')

    ax2.plot(MoSe2_DOS[:,1], MoSe2_DOS[:,0], color='g')
    ax2.plot(WSe2_DOS[:,1], WSe2_DOS[:,0], color='b')
    ax2.plot(Hetero_DOS[:,1], Hetero_DOS[:,0], color='k')
    ax2.set_xlim(0, 2.0)
    ax2.set_xticks(np.arange(0, 2.3,1.0))
    ax2.set_yticks([])
    ax2.set_xlabel('density (arb.)', labelpad=2)

    for ax in [ax1, ax2]:
        thermal_scale(ax)
        ax.set_ylim(0, 40)

    ax1.set_ylim(0, 40)
# end plot_phonon_dispersion

def convert(meV):
    return 1.0/(1e-3*meV)
# end convert

def DOS_inverse_E(ax, DOS):
    WSe2_DOS, MoSe2_DOS, Hetero_DOS = DOS
    ax.plot(MoSe2_DOS[:,1], convert(MoSe2_DOS[:,0]), color='g')
    ax.plot(WSe2_DOS[:,1], convert(WSe2_DOS[:,0]), color='b')
    ax.plot(Hetero_DOS[:,1], convert(Hetero_DOS[:,0]), color='k')
    ax.set_ylim(10, 90)
    ax.set_xlim(0, 2)
    ax.set_ylabel('1/eV')
    ax.set_xlabel('density (arb.)')
# end DOS_inverse_E

if __name__ == '__main__':
    save, svfile = display.argsave()

    bands = load_dispersion()

    WSe2_DOS = load_DOS("WSe2_phonon_DOS")
    MoSe2_DOS = load_DOS("MoSe2_phonon_DOS")
    Hetero_DOS = load_DOS("Hetero_phonon_DOS")
    DOS = [WSe2_DOS, MoSe2_DOS, Hetero_DOS]

    display.paper_figure_format(labelpad=5)

    xinches = 6.0
    yinches = 4.0
    fi = display.figure_inches(__file__, xinches, yinches)
    xmargin = 0.65
    ymargin = 0.5

    width2 = 1.6
    xint = 0.2
    yint = 0.75
    width = 2*width2 + xint
    height = width


    cbwidth = 0.06*width2
    cbheight = 0.33*height
    cbmargin = 0.03*width2

    ax11 = fi.make_axes([xmargin, ymargin, width, height])
    ax12 = fi.make_axes([xmargin+width+xint, ymargin, width2, height])

    plot_phonon_dispersion(ax11, ax12, bands, DOS)

    if save:
        plt.savefig(join(svfile, 'figS31.png'), dpi=300)

    plt.show()
