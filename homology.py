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

##### Ref: https://colab.research.google.com/drive/1addhGqN3ZE1mIn4L6jQnnkVs7_y__qSE?usp=sharing
##### Credit: Florent Poux
def grid_subsampling(points, voxel_size):

  nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0))/voxel_size)
  non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
  idx_pts_vox_sorted = np.argsort(inverse)
  voxel_grid = {}
  grid_barycenter,grid_candidate_center = [],[]
  last_seen = 0

  for idx,vox in enumerate(non_empty_voxel_keys):
    voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
    grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
    grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)]-np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
    last_seen += nb_pts_per_voxel[idx]

  return grid_candidate_center
    
### params

cell_type = "Iba1"
max_dim = 1
grid_voxel_size = 150

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
        
        #### load and sample point cloud
    
        df = pd.read_csv(csv_fname)
        coords = df[['X', 'Y', 'Z']].to_numpy()
        pcd = np.array(grid_subsampling(coords, grid_voxel_size))
        
        print("Subsampled " + str(pcd.shape[0]) + " out of " + str(coords.shape[0]) + " points")

        dgm = rpp_py.run("--dim " + str(max_dim) + " --format point-cloud", pcd)
        gtda_dgm = ripser2gtda(dgm, max_dim)
        
        bc = betti_curve.fit_transform([gtda_dgm])
        
        TDA_data.append(copy.deepcopy((dpi, mid, gtda_dgm, bc, pcd)))
        
        if not os.path.exists(os.path.join("results", cell_type)):
            os.makedirs(os.path.join("results", cell_type))
        ofilename = mid + "_" + dpi + "_" + ctype + ".npy"
        np.save(os.path.join("results", cell_type, ofilename), {"dgm":gtda_dgm, "bc":bc, "data":pcd})    

### plotting

for dim in range(max_dim+1):
    
    plt.figure(figsize=(10,3), dpi=200)

    for idx in range(len(TDA_data)):
        
        (dpi, mid, pers_diags, betti_curves, pt_cloud) = TDA_data[idx]

        bc = betti_curves[0][dim]
        num_samples = pt_cloud.shape[0]
        
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