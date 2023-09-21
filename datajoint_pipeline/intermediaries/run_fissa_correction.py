import fissa
import numpy as np
import os
from natsort import natsorted

def run_fissa_correction(plane_path, sampling_freq):
    # plane_path = r'\\nerffs17\boninlabwip2023\data\2photon\reg\230411_KC143_AL__2P_KC_s2p_04-09-2023\suite2p\plane0'
    images = os.path.join(plane_path, "reg_tif")

    tiff_paths = [os.path.join(images, f.name) for f in os.scandir(images)]
    ordered_tiffs = natsorted(tiff_paths)

    # Load the detected regions of interest
    stat = np.load(os.path.join(plane_path, "stat.npy"), allow_pickle=True)  # cell stats
    ops = np.load(os.path.join(plane_path, "ops.npy"), allow_pickle=True).item()
    iscell = np.load(os.path.join(plane_path, "iscell.npy"), allow_pickle=True)[:, 0]

    # Get image size
    Lx = ops["Lx"]
    Ly = ops["Ly"]

    # Get the cell ids
    ncells = len(stat)
    cell_ids = np.arange(ncells)  # assign each cell an ID, starting from 0.
    cell_ids = cell_ids[iscell == 1]  # only take the ROIs that are actually cells.
    num_rois = len(cell_ids)

    # Generate ROI masks in a format usable by FISSA (in this case, a list of masks)
    rois = [np.zeros((Ly, Lx), dtype=bool) for n in range(num_rois)]

    for i, n in enumerate(cell_ids):
        # i is the position in cell_ids, and n is the actual cell number
        ypix = stat[n]["ypix"][~stat[n]["overlap"]]
        xpix = stat[n]["xpix"][~stat[n]["overlap"]]
        rois[i][ypix, xpix] = 1

    output_folder = os.path.join(plane_path, "fissa_output")
    experiment = fissa.Experiment(ordered_tiffs, [rois[:ncells]], output_folder, verbosity=1, ncores_separation=8)
    experiment.separate()
    experiment.calc_deltaf(freq=sampling_freq, use_raw_f0=False, across_trials=True)

    return experiment, rois
