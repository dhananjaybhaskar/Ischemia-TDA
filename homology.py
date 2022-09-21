import os, glob, copy, random

import numpy as np
import pandas as pd

import ripserplusplus as rpp_py

from gtda.pipeline import Pipeline
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

### helper methods

def ripser2gtda(dgm, max_dim):
    
    diags = []
    for dim in range(max_dim+1):
        num_pts = len(dgm[dim])
        pers_diag = np.zeros((num_pts, 3))
        for idx in range(num_pts):
            pers_diag[idx,0] = dgm[dim][idx][0]
            pers_diag[idx,1] = dgm[dim][idx][1]
            pers_diag[idx,2] = dim
        diags.append(copy.deepcopy(pers_diag))
    return(np.vstack(diags))    
    
### params

cell_type = "Gfap"
max_dim = 1
num_samples = 10000

csv_files = glob.glob(os.path.join("data", cell_type, "*.csv"))
betti_curve = BettiCurve(n_bins=100, n_jobs=-1)

TDA_data = []

for csv_fname in csv_files:
    
    fname_parts = os.path.splitext(os.path.basename(csv_fname))[0].split("_")
    ctype = fname_parts[-1]
    dpi = fname_parts[-2]
    mid = fname_parts[-3]
    
    if ctype == cell_type:
        
        print(csv_fname + ":\t MouseID: " + mid[1:] + "\t Days Post Injury: " + dpi[:-1])
    
        df = pd.read_csv(csv_fname)
        coords = df[['X', 'Y', 'Z']].to_numpy()
        np.random.shuffle(coords)

        dgm = rpp_py.run("--dim " + str(max_dim) + " --format point-cloud", coords[:num_samples, :])
        gtda_dgm = ripser2gtda(dgm, max_dim)
        
        bc = betti_curve.fit_transform([gtda_dgm])
        
        TDA_data.append(copy.deepcopy((dpi, mid, gtda_dgm, bc)))
        

### plotting

for dim in range(max_dim+1):
    
    plt.figure(figsize=(10,3), dpi=200)

    for idx in range(len(TDA_data)):
        
        (dpi, mid, pers_diags, betti_curves) = TDA_data[idx]

        bc = betti_curves[0][dim]
        
        if dpi == "0D":
            plt.plot(bc.flatten()/num_samples, linewidth=0.4, alpha=0.7, color="indigo")
        elif dpi == "5D":
            plt.plot(bc.flatten()/num_samples, linewidth=0.4, alpha=0.7, color="darkred")
        elif dpi == "15D":
            plt.plot(bc.flatten()/num_samples, linewidth=0.4, alpha=0.7, color="darkgreen")
        elif dpi == "30D":
            plt.plot(bc.flatten()/num_samples, linewidth=0.4, alpha=0.7, color="gold")
        else:
            print("Unknown DPI: " + ctype)
            
    plt.plot(np.NaN, np.NaN, '-', color='indigo', label='0D')
    plt.plot(np.NaN, np.NaN, '-', color='darkred', label='5D')
    plt.plot(np.NaN, np.NaN, '-', color='darkgreen', label='15D')
    plt.plot(np.NaN, np.NaN, '-', color='gold', label='30D')

    plt.xlabel("eps", fontsize=10)
    plt.ylabel("betti fraction", fontsize=10)
    plt.title("H" + str(dim) + " Betti Curve (" + cell_type + ")")

    leg = plt.legend(frameon=False)
    leg.get_frame().set_facecolor('none')

    plt.savefig(cell_type + "_H" + str(dim) + "_betticurve" + ".png")