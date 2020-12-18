'''
imaging_datasets.py

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
Module for loading the MPDPM imaging datasets from the paper 'Stacking enabled
strong coupling of atomic motion to interlayer excitons in van der Waals
heterojunction photodiodes'

See accompanying README.txt for instructions on using this code.
'''

import numpy as np
from scipy.ndimage import gaussian_filter
from os.path import join, exists
from matplotlib.path import Path

# Takes the average of points from data that are within the path (matplotlib path)
# Wich has indicies of the path ax X and Y coordinates
def avg_within_path(data, path):
    rows, cols = data.shape
    N = 0
    sum = 0.0
    for i in range(rows):
        for j in range(cols):
            if path.contains_point((j, i)):
                sum += data[i,j]
                N += 1
    return sum/N
# end avg_within_path

def load_image_dataset(key="Vg_Vsd_narrow", refdata=False):
    savefile = join("datasets",'mpdpm-imaging-sets')
    files = np.load(join(savefile,key+".npz"))
    Vsd = files['Vsd'] # Source/Drain voltage (mV)
    Vg = files['Vg'] # Gate Voltage (V)
    d = files['d'] # drift corrected photocurrent images (nA)
    if refdata:
        r = np.load(join(savefile,key+"_ref.npy")) # drift corrected Reflection (arb.)

    rows, cols, Mg, Mb = d.shape
    Vsd = Vsd*1e-3 # Convert to V

    hetero_file = join('datasets','mpdpm-imaging-sets','area_averaged',key+'_hetero.npy')
    if exists(hetero_file):
        heterostructure = np.load(hetero_file)
    else:
        print("Processing heterostructure data, this may take a few minutes")
        # Heterostructure Region
        if key == "Vg_Vsd_narrow":
            X = np.array([36, 30, 30, 37, 36])
            Y = np.array([54, 58, 47, 43, 54])
        else:
            raise ValueError("Invalid Dataset for spatial averaging")
        verts = np.zeros((np.size(X),2))
        verts[:,0] = X
        verts[:,1] = Y
        heteroarea = Path(verts)

        # Re-orient and re-arange so that is in the format that it is plotted in
        avgd = np.zeros((Mb, Mg))
        for i in range(Mg):
            for j in range(Mb):
                avgd[j,i] = avg_within_path(np.rot90(d[:,:,i,j]), heteroarea)
        #

        heterostructure = gaussian_filter(avgd,1.0)
        np.save(hetero_file, heterostructure)

    if refdata:
        return Vsd, Vg, d, heterostructure, r
    else:
        return Vsd, Vg, d, heterostructure
# end plot_phase_map

def load_spectro_image_dataset(key="Vg_Wave_350"):
    savefile = join("datasets",'mpdpm-imaging-sets',key+".npz")
    if exists(savefile):
        files = np.load(savefile)
        Vsd = files['Vsd']
        Vg = files['Vg']
        w = files['w']
        pw = files['pw']
        d = files['d']
        return Vsd, Vg, w, pw, d
    else:
        raise ValueError("Invalid dataset name " + key)
# end load_spectro_image_dataset
