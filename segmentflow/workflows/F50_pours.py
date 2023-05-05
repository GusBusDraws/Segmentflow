import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view, mesh
from stl import mesh
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow is the  to segment particles in a CT scan'
    ' according to the preferences set in an input YAML file.'
    ' Output can be a labeled TIF stack and/or STL files corresponding'
    ' to each segmented particle.'
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
    'stl_overwrite'        : False,
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
    ui = segment.process_args(
        argv, WORKFLOW_NAME, WORKFLOW_DESCRIPTION, CATEGORIZED_INPUT_SHORTHANDS,
        DEFAULT_VALUES
    )

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

    #-------------------#
    # Preprocess images #
    #-------------------#
    print()
    imgs_pre = segment.preprocess(
        imgs, median_filter=ui['pre_seg_med_filter'],
        rescale_intensity_range=ui['rescale_range']
    )

    #-----------------------#
    # Semantic segmentation #
    #-----------------------#
    print()
    if ui['view_thresh_hist']:
        thresholds, thresh_fig, thresh_ax = segment.threshold_multi_min(
            imgs_pre, nbins=ui['thresh_nbins'], return_fig_ax=True,
            ylims=ui['thresh_hist_ylims']
        )
    else:
        thresholds = segment.threshold_multi_min(
            imgs_pre, nbins=ui['thresh_nbins'], return_fig_ax=False,
        )
    imgs_semantic = segment.isolate_classes(imgs_pre, thresholds)
    if ui['view_semantic']:
        fig, axes = view.plot_slices(
                imgs_semantic,
                slices=ui['view_slices'],
                print_slices=False,
                fig_w=7.5,
                dpi=100
            )

    #-----------------------#
    # Instance segmentation #
    #-----------------------#
    if ui['perform_seg']:
        print()
        imgs = None
        imgs_pre = None
        imgs_labeled = segment.watershed_segment(
            imgs_semantic==len(thresholds),
            min_peak_distance=ui['min_peak_dist'],
            exclude_borders=ui['exclude_borders'],
            return_dict=False
        )
        # Merge semantic and instance segmentations
        imgs_labeled = segment.merge_segmentations(imgs_semantic, imgs_labeled)
        if ui['view_labeled']:
            fig, axes = view.plot_color_labels(
                imgs_labeled,
                slices=ui['view_slices'],
                fig_w=7.5,
                dpi=100
            )
        if ui['save_voxels']:
            segment.save_images(
                imgs_labeled,
                Path(ui['out_dir_path']) / f"{ui['out_prefix']}_labeled_voxels"
            )

    #----------------------------------------#
    # Create Surface Meshes of Each Particle #
    #----------------------------------------#
    if ui['perform_seg'] and ui['create_stls']:
        print()
        segment.save_as_stl_files(
            imgs_labeled,
            ui['out_dir_path'],
            ui['out_prefix'],
            suppress_save_msg=ui['suppress_save_msg'],
            slice_crop=ui['slice_crop'],
            row_crop=ui['row_crop'],
            col_crop=ui['col_crop'],
            stl_overwrite=ui['overwrite'],
            spatial_res=ui['spatial_res'],
            n_erosions=ui['n_erosions'],
            median_filter_voxels=ui['post_seg_med_filter'],
            voxel_step_size=ui['voxel_step_size'],
        )

        #----------------------------------------------#
        # Postprocess surface meshes for each particle #
        #----------------------------------------------#
        if (
            ui['mesh_smooth_n_iters'] is not None
            or ui['mesh_simplify_n_tris'] is not None
            or ui['mesh_simplify_factor'] is not None
        ):
            print()
            # Iterate through each STL file, load the mesh, and smooth/simplify
            mesh.postprocess_meshes(
                ui['stl_dir_location'],
                smooth_iter=ui['mesh_smooth_n_iters'],
                simplify_n_tris=ui['mesh_simplify_n_tris'],
                iterative_simplify_factor=ui['mesh_simplify_factor'],
                recursive_simplify=False, resave_mesh=True
            )

    #-------------------------#
    # Plot figures if enabled #
    #-------------------------#
    if (
        ui['view_raw'] or ui['view_pre'] or ui['view_semantic']
        or ui['view_labeled']
    ):
        plt.show()


if __name__ == '__main__':
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    print(f'Beginning Workflow {WORKFLOW_NAME}')
    print()
    workflow(sys.argv[1:])
    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Successful Completion')
    print('~~~~~~~~~~~~~~~~~~~~~')
    print()

