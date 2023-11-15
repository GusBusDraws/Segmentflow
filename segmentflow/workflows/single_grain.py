import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from scipy import ndimage as ndi
from segmentflow import segment, view, mesh
from skimage import filters, measure, morphology
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow segments a single particle from a synchrotron CT scan of a'
    ' single F50 sand grain. Outputs can be a labeled TIF stack and/or STL'
    ' file. Developed for Segmentflow v0.0.3.'
)

CATEGORIZED_INPUT_SHORTHANDS = {
    'A. Input' : {
        'in_dir_path'  : '01. Input dir path',
        'file_suffix'  : '02. File suffix',
        'slice_crop'   : '03. Slice crop',
        'row_crop'     : '04. Row crop',
        'col_crop'     : '05. Column crop',
        'spatial_res'  : '06. Pixel size',
    },
    'B. Processing' : {
        'min_size_keep' : '01. Minimum volume of regions to keep',
        'ero_dil_iters' : '02. Number of erosion-dilation iterations',
        'med_filt_size' : '03. Diameter of median filter (odd number, pixels)',
        'flip_z'        : '04. Flip voxels in z direction',
        'mesh_step'     : '05. Voxel step size in surface mesh creation',
    },
    'C. Output' : {
        'overwrite'    : '01. Overwrite files',
        'out_prefix'   : '02. Output prefix',
        'nslices'      : '03. Number of slices in checkpoint plots',
        'out_dir_path' : '04. Path to save output dir',
        'save_stl'     : '05. Save STL file',
        'save_voxels'  : '06. Save voxel TIF stack'
    },
}

DEFAULT_VALUES = {
    'in_dir_path'   : 'REQUIRED',
    'file_suffix'   : '.tif',
    'slice_crop'    : None,
    'row_crop'      : None,
    'col_crop'      : None,
    'spatial_res'   : 1,
    'min_size_keep' : 500,
    'med_filt_size' : 3,
    'ero_dil_iters' : 8,
    'flip_z'        : False,
    'mesh_step'     : 1,
    'out_dir_path'  : 'REQUIRED',
    'out_prefix'    : '',
    'overwrite'     : True,
    'nslices'       : 5,
    'save_stl'      : True,
    'save_voxels'   : True,
}

#~~~~~~~~~~#
# Workflow #
#~~~~~~~~~~#
def workflow(argv):
    #-----------------------------------------------------#
    # Get command-line arguments and read YAML input file #
    #-----------------------------------------------------#
    ui = segment.process_args(
        argv, WORKFLOW_NAME, WORKFLOW_DESCRIPTION, CATEGORIZED_INPUT_SHORTHANDS,
        DEFAULT_VALUES
    )

    n_fig_digits = 2
    fig_n = 0
    if not Path(ui['out_dir_path']).exists():
        Path(ui['out_dir_path']).mkdir(parents=True)
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
        dpi=300
    )
    plt.savefig(
        Path(ui['out_dir_path'])
        / f'{str(fig_n).zfill(n_fig_digits)}-raw-imgs.png')

    #-------------------#
    # Preprocess images #
    #-------------------#
    print()
    # Median filtering to reduce noise
    imgs_med = segment.preprocess(
        imgs, median_filter=True,
        rescale_intensity_range=None
    )

    #-----------------------#
    # Semantic segmentation #
    #-----------------------#
    imgs_binarized, thresh_vals = segment.binarize_multiotsu(
        imgs_med, n_otsu_classes=2
    )
    # Plot histogram
    hist, bin_edges = np.histogram(imgs_med)
    fig, ax = plt.subplots(dpi=300)
    ax.plot(bin_edges[1:], hist, lw=1)
    for val in thresh_vals:
        ax.axvline(val, c='red', zorder=0)
    fig_n += 1
    plt.savefig(
        Path(ui['out_dir_path'])
        / f'{str(fig_n).zfill(n_fig_digits)}-hist.png')
    # zyx
    fig, axes = view.plot_slices(
        imgs_binarized,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=300
    )
    fig_n += 1
    plt.savefig(
        Path(ui['out_dir_path'])
        / f'{str(fig_n).zfill(n_fig_digits)}-bw-yx.png')
    # Plot rotated crop - yxz
    imgs_binarized_xz = np.rot90(imgs_binarized, axes=(2, 0))
    fig, axes = view.plot_slices(
        imgs_binarized_xz,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=300
    )
    fig_n += 1
    plt.savefig(
        Path(ui['out_dir_path'])
        / f'{str(fig_n).zfill(n_fig_digits)}-bw-xz.png')

    #-------------------------------------------#
    # Remove small particles outside sand grain #
    #-------------------------------------------#
    print(f"Clearing noise smaller than {ui['min_size_keep']} voxels...")
    imgs_cleaned = np.zeros_like(imgs_binarized)
    for n in range(imgs_binarized.shape[0]):
        imgs_cleaned[n, ...] = morphology.remove_small_objects(
            measure.label(imgs_binarized[n, ...]),
            min_size=ui['min_size_keep']).astype(bool)
    # Fill small holes inside sand grain
    imgs_filled = np.zeros_like(imgs_cleaned)
    for n in range(imgs_cleaned.shape[0]):
        imgs_filled[n, ...] = ndi.binary_fill_holes(imgs_cleaned[n, ...])
    # Plot filled particle
    fig, axes = view.plot_slices(
        imgs_filled,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=300
    )
    fig_n += 1
    plt.savefig(
        Path(ui['out_dir_path'])
        / f'{str(fig_n).zfill(n_fig_digits)}-filled.png')

    #---------------------------------#
    # Remove reconstruction artifacts #
    #---------------------------------#
    # Ring-like artifacts extruding from edge of sand grain:
    # Attempt to remove by eroding and then "re-roding" (dilating) region
    print('Removing reconstruction artifacts...')
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
        imgs_eroded = imgs_largest_only
    # Plot eroded particle
    # zyx
    fig, axes = view.plot_slices(
        imgs_eroded,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=300
    )
    fig_n += 1
    plt.savefig(
        Path(ui['out_dir_path'])
        / f'{str(fig_n).zfill(n_fig_digits)}-eroded-yx.png')
    # yxz
    imgs_eroded_xz = np.rot90(imgs_eroded, axes=(2, 0))
    fig, axes = view.plot_slices(
        imgs_eroded_xz,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=300
    )
    fig_n += 1
    plt.savefig(
        Path(ui['out_dir_path'])
        / f'{str(fig_n).zfill(n_fig_digits)}-eroded-xz.png')
    #---------------#
    # Smooth voxels #
    #---------------#
    if (ui['med_filt_size'] % 2) == 0:
        ui['med_filt_size'] -= 1
        print(
            'Median filter size is not an odd number.'
            f"Setting to {ui['med_filt_size']}"
        )
    if (ui['med_filt_size']) > 1:
        print('Applying post-processing median filter...')
        imgs_eroded = filters.median(
            imgs_filled_labeled,
            footprint=morphology.ball(ui['med_filt_size'])
        )
        # Plot median filtered
        # zyx
        fig, axes = view.plot_slices(
            imgs_eroded,
            nslices=ui['nslices'],
            fig_w=7.5,
            dpi=300
        )
        fig_n += 1
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-median-filtered-yx.png')
        # yxz
        imgs_eroded_xz = np.rot90(imgs_eroded, axes=(2, 0))
        fig, axes = view.plot_slices(
            imgs_eroded_xz,
            nslices=ui['nslices'],
            fig_w=7.5,
            dpi=300
        )
        fig_n += 1
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-median-filtered-xz.png')

    #----------------------------#
    # Flip voxels in z-direction #
    #----------------------------#
    if ui['flip_z']:
        print('FLipping voxels in z direction...')
        imgs_filled_labeled = imgs_filled_labeled[::-1]
        # Plot eroded particle
        # zyx
        fig, axes = view.plot_slices(
            imgs_filled_labeled,
            nslices=ui['nslices'],
            fig_w=7.5,
            dpi=300
        )
        fig_n += 1
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-z_flipped-yx.png')
        # yxz
        imgs_filled_labeled_xz = np.rot90(imgs_filled_labeled, axes=(2, 0))
        fig, axes = view.plot_slices(
            imgs_filled_labeled_xz,
            nslices=ui['nslices'],
            fig_w=7.5,
            dpi=300
        )
        fig_n += 1
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-z_flipped-xz.png')

    #--------------#
    # Save outputs #
    #--------------#
    # Save largest particle as STL
    # Resolution = 1.09 micrometers per pixel (0.00109 mm per pixel)
    if ui['save_stl']:
        segment.save_as_stl_files(
            imgs_filled_labeled,
            ui['out_dir_path'],
            ui['out_prefix'],
            n_erosions=1,
            median_filter_voxels=True,
            voxel_step_size=ui['mesh_step'],
            make_new_save_dir=True,
            spatial_res=ui['spatial_res'],
            stl_overwrite=ui['overwrite']
        )
    # save images
    if ui['save_voxels']:
        imgs_labeled_8bit = imgs_filled_labeled.astype(np.ubyte)
        imgs_labeled_8bit[imgs_labeled_8bit == 1] = 255
        segment.save_images(
            imgs_labeled_8bit,
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

