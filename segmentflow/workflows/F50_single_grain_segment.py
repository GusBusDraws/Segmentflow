import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import ndimage as ndi
from segmentflow import segment, view, mesh
from skimage import measure, morphology
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow segments a single particle from a synchrotron CT scan of a'
    ' single F50 sand grain. Outputs can be a labeled TIF stack and/or STL'
    ' file. Developed for Segmentflow v0.0.3.'
)

CATEGORIZED_INPUT_SHORTHANDS = {
    'Files' : {
        'in_dir_path'  : 'Input dir path',
        'file_suffix'  : 'File suffix',
        'slice_crop'   : 'Slice crop',
        'row_crop'     : 'Row crop',
        'col_crop'     : 'Column crop',
        'spatial_res'  : 'Pixel size',
        'out_dir_path' : 'Path to save output dir',
        'out_prefix'   : 'Output prefix',
        'overwrite'    : 'Overwrite files'
    },
    'View' : {
        'view_slices'   : 'Slices to view',
        'view_raw'      : 'View raw images',
        'view_pre'      : 'View preprocessed images',
        'view_semantic' : 'View semantic images',
        'view_labeled'  : 'View labeled images',
    },
    'Preprocess' : {
        'pre_seg_med_filter' : 'Apply median filter',
        'rescale_range'      : 'Rescale intensity range',
    },
    'Segmentation' : {
        'thresh_nbins'      : 'Histogram bins for calculating thresholds',
        'view_thresh_hist'  : 'View histogram with threshold values',
        'thresh_hist_ylims' : 'Upper and lower y-limits of histogram',
        'perform_seg'       : 'Perform instance segmentation',
        'min_peak_dist'     : 'Min peak distance',
        'exclude_borders'   : 'Exclude border particles',
        'save_voxels'       : 'Save labeled voxels'
    },
    'STL' : {
        'create_stls'          : 'Create STL files',
        'suppress_save_msg'    : 'Suppress save message for each STL file',
        'n_erosions'           : 'Number of pre-surface meshing erosions',
        'post_seg_med_filter'  : 'Smooth voxels with median filtering',
        'voxel_step_size'      : 'Marching cubes voxel step size',
        'mesh_smooth_n_iters'  : 'Number of smoothing iterations',
        'mesh_simplify_n_tris' : 'Target number of triangles/faces',
        'mesh_simplify_factor' : 'Simplification factor per iteration',
    },
}

DEFAULT_VALUES = {
    'in_dir_path'          : 'REQUIRED',
    'file_suffix'          : 'tiff',
    'slice_crop'           : None,
    'row_crop'             : None,
    'col_crop'             : None,
    'out_dir_path'         : 'REQUIRED',
    'out_prefix'           : '',
    'overwrite'            : False,
    'view_slices'          : True,
    'view_raw'             : True,
    'view_pre'             : True,
    'view_semantic'        : True,
    'view_labeled'         : True,
    'pre_seg_med_filter'   : False,
    'rescale_range'        : None,
    'thresh_nbins'         : 256,
    'view_thresh_hist'     : True,
    'thresh_hist_ylims'    : [0, 2e7],
    'perform_seg'          : True,
    'min_peak_dist'        : 6,
    'exclude_borders'      : True,
    'save_voxels'          : True,
    'create_stls'          : True,
    'suppress_save_msg'    : True,
    'n_erosions'           : 0,
    'post_seg_med_filter'  : False,
    'spatial_res'          : 1,
    'voxel_step_size'      : 1,
    'mesh_smooth_n_iters'  : None,
    'mesh_simplify_n_tris' : None,
    'mesh_simplify_factor' : None,
    'seg_fig_show'         : False,
}

#~~~~~~~~~~#
# Workflow #
#~~~~~~~~~~#
def workflow(argv):
    #-----------------------------------------------------#
    # Get command-line arguments and read YAML input file #
    #-----------------------------------------------------#
    # ui = segment.process_args(
    #     argv, WORKFLOW_NAME, WORKFLOW_DESCRIPTION, CATEGORIZED_INPUT_SHORTHANDS,
    #     DEFAULT_VALUES
    # )
    ui = {}
    ui['in_dir_path'] = r'C:\Users\gusb\Research\mhe-analysis\data\F50_1_Scan_1'
    ui['file_suffix'] = '.tif'
    ui['slice_crop'] = [340, 600]
    ui['row_crop'] = [600, 1250]
    ui['col_crop'] = [600, 1250]
    ui['out_dir_path'] = r'C:\Users\gusb\Research\mhe-analysis\results\F50_1_Scan_1'
    ui['out_prefix'] = 'F50_1_Scan_1'
    ui['overwrite'] = True
    ui['spatial_res'] = 0.00109
    ui['nslices'] = 4
    ui['rm_min_size'] = 500
    ui['ero_dil_iters'] = 8

    #-------------#
    # Load images #
    #-------------#
    print()
    imgs = segment.load_images(
        ui['in_dir_path'],
        slice_crop=ui['slice_crop'],
        row_crop=ui['row_crop'],
        col_crop=ui['col_crop'],
        convert_to_float=True,
        file_suffix=ui['file_suffix']
    )
    fig, axes = view.plot_slices(
        imgs,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=100
    )
    plt.show()

    #-------------------#
    # Preprocess images #
    #-------------------#
    print()
    # Median filtering to reduce noise
    imgs_med = segment.preprocess(
        imgs, median_filter=True,
        rescale_intensity_range=None
    )
    # Plot histogram
    hist, bin_edges = np.histogram(imgs_med)
    fig, ax = plt.subplots(dpi=150)
    ax.plot(bin_edges[1:], hist, lw=1)
    for val in thresh_vals:
        ax.axvline(val, c='red', zorder=0)
    plt.show()

    #-----------------------#
    # Semantic segmentation #
    #-----------------------#
    imgs_binarized, thresh_vals = segment.binarize_multiotsu(
        imgs_med, n_otsu_classes=2
    )
    # zyx
    fig, axes = view.plot_slices(
        imgs_binarized,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=100
    )
    # Plot rotated crop - yxz
    imgs_binarized_xz = np.rot90(imgs_binarized, axes=(2, 0))
    print(f'{imgs_binarized_xz.shape=}')
    fig, axes = view.plot_slices(
        imgs_binarized_xz,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=100
    )
    plt.show()

    #-------------------------------------------#
    # Remove small particles outside sand grain #
    #-------------------------------------------#
    imgs_cleaned = np.zeros_like(imgs_binarized)
    for n in range(imgs_binarized.shape[0]):
        imgs_cleaned[n, ...] = morphology.remove_small_objects(
            measure.label(imgs_binarized[n, ...]),
            min_size=ui['rm_min_size']).astype(bool)
    # Fill small holes inside sand grain
    imgs_filled = np.zeros_like(imgs_cleaned)
    for n in range(imgs_cleaned.shape[0]):
        imgs_filled[n, ...] = ndi.binary_fill_holes(imgs_cleaned[n, ...])
    # Plot images with noise cleaned
    fig, axes = view.plot_slices(
        imgs_cleaned,
        nslices=ui['nslices'],
        imgs_per_row=4,
        fig_w=7.5,
        dpi=100
    )
    plt.show()
    # Plot filled particle
    fig, axes = view.plot_slices(
        imgs_filled,
        nslices=ui['nslices'],
        imgs_per_row=4,
        fig_w=7.5,
        dpi=100
    )
    plt.show()

    #---------------------------------#
    # Remove reconstruction artifacts #
    #---------------------------------#
    # Ring-like artifacts extruding from edge of sand grain:
    # Attempt to remove by eroding and then "re-roding" (dilating) region
    imgs_eroded = ndi.binary_erosion(
        imgs_filled,
        iterations=ui['ero_dil_iters']
    )
    imgs_eroded = ndi.binary_dilation(
        imgs_eroded,
        iterations=ui['ero_dil_iters']
    )
    # If max value of labeled images is not 1, there is more than one connected
    # particle and the largest needs to be isolated from the rest
    imgs_filled_labeled = measure.label(imgs_eroded)
    print('Number of particles =', imgs_filled_labeled.max())
    if imgs_filled_labeled.max() > 1:
        print('Isolating the largest particle...')
        df = pd.DataFrame(measure.regionprops_table(
            imgs_filled_labeled, properties=['label', 'area', 'bbox']
        ))
        df = df.rename(columns={'area' : 'volume'})
        # Get the label according to the particle with the largest volume
        largest_label = df.loc[df.volume.idxmax(), 'label']
        imgs_largest_only = np.zeros_like(imgs_filled_labeled, dtype=np.ubyte)
        imgs_largest_only[imgs_filled_labeled == largest_label] = 1
        imgs_filled_labeled = imgs_largest_only
    # Plot eroded particle
    # zyx
    fig, axes = view.plot_slices(
        imgs_eroded,
        nslices=ui['nslices'],
        imgs_per_row=4,
        fig_w=7.5,
        dpi=100
    )
    # yxz
    imgs_eroded_xz = np.rot90(imgs_eroded, axes=(2, 0)),
    fig, axes = view.plot_slices(
        imgs_eroded_xz,
        nslices=ui['nslices'],
        imgs_per_row=4,
        fig_w=7.5,
        dpi=100
    )
    plt.show()

    #--------------#
    # Save outputs #
    #--------------#
    # Save largest particle as STL - Resolution = 1.09 micrometers per pixel (0.00109 mm per pixel)
    if ui['save_stl']:
        segment.save_as_stl_files(
            imgs_filled_labeled,
            ui['out_dir_path'],
            ui['out_prefix'],
            make_new_save_dir=False,
            spatial_res=ui['spatial_res'],
            stl_overwrite=ui['overwrite']
        )
    # save images
    if ui['save_voxels']:
        segment.save_images(
            imgs_filled_labeled,
            Path(ui['out_dir_path']) / f"{ui['out_prefix']}_labeled_voxels"
        )


if __name__ == '__main__':
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    print(f'Beginning workflow: {WORKFLOW_NAME}')
    print()
    workflow(sys.argv[1:])
    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Successful Completion')
    print('~~~~~~~~~~~~~~~~~~~~~')
    print()

