import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view, mesh
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow segments IDOX particles and Kel-F binder in a CT scan'
    ' and outputs a labeled TIF stack and/or STL files corresponding'
    ' to each segmented particle. Developed for v0.0.1.'
)

CATEGORIZED_INPUT_SHORTHANDS = {
    'A. Files' : {
        'in_dir_path'  : '01. Input dir path',
        'file_suffix'  : '02. File suffix',
        'slice_crop'   : '03. Slice crop',
        'row_crop'     : '04. Row crop',
        'col_crop'     : '05. Column crop',
        'spatial_res'  : '06. Pixel size',
        'out_dir_path' : '07. Path to save output dir',
        'out_prefix'   : '08. Output prefix',
        'overwrite'    : '09. Overwrite files'
    },
    'B. Preprocess' : {
        'pre_seg_med_filter' : '01. Apply median filter',
        'rescale_range'      : '02. Rescale intensity range',
    },
    'C. Segmentation' : {
        'thresh_nbins'      : '01. Histogram bins for calculating thresholds',
        'thresh_hist_ylims' : '02. Upper and lower y-limits of histogram',
        'perform_seg'       : '03. Perform instance segmentation',
        'min_peak_dist'     : '04. Min peak distance',
        'exclude_borders'   : '05. Exclude border particles',
    },
    'D. Output' : {
        'save_voxels'          : '01. Save labeled voxels',
        'nslices'              : '02. Number of slices in checkpoint plots',
        'slices'               : '03. Specific slices to plot',
        'save_stls'            : '04. Create STL files',
        'suppress_save_msg'    : '05. Suppress save message for each STL file',
        'n_erosions'           : '06. Number of pre-surface meshing erosions',
        'post_seg_med_filter'  : '07. Smooth voxels with median filtering',
        'step_size'            : '08. Marching cubes voxel step size',
        'mesh_smooth_n_iters'  : '09. Number of smoothing iterations',
        'mesh_simplify_n_tris' : '10. Target number of triangles/faces',
        'mesh_simplify_factor' : '11. Simplification factor per iteration',
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
    'pre_seg_med_filter'   : False,
    'rescale_range'        : None,
    'thresh_nbins'         : 256,
    'thresh_hist_ylims'    : [0, 20000000],
    'perform_seg'          : True,
    'min_peak_dist'        : 6,
    'exclude_borders'      : True,
    'save_voxels'          : True,
    'nslices'              : 3,
    'slices'               : None,
    'save_stls'            : True,
    'n_erosions'           : 1,
    'suppress_save_msg'    : True,
    'post_seg_med_filter'  : False,
    'spatial_res'          : 1,
    'step_size'            : 1,
    'mesh_smooth_n_iters'  : None,
    'mesh_simplify_n_tris' : None,
    'mesh_simplify_factor' : None,
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

    show_checkpoints = False
    checkpoint_save_dir = ui['out_dir_path']
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
    # Generate raw imgs viz
    fig, axes = view.vol_slices(
            imgs,
            slices=ui['slices'],
            nslices=ui['nslices'],
            print_slices=False,
            fig_w=7.5,
            dpi=300
        )
    fig_n = 0
    segment.output_checkpoints(
        fig, show=show_checkpoints, save_path=checkpoint_save_dir,
        fn_n=fig_n, fn_suffix='raw-imgs')

    #-------------------#
    # Preprocess images #
    #-------------------#
    print()
    # Plot intensity rescale histogram
    imgs_med = segment.preprocess(
        imgs, median_filter=ui['pre_seg_med_filter'])
    fig, ax = view.histogram(imgs_med, mark_percentiles=ui['rescale_range'])
    fig_n += 1
    segment.output_checkpoints(
        fig, show=show_checkpoints, save_path=checkpoint_save_dir,
        fn_n=fig_n, fn_suffix='intensity-rescale-hist')
    # Preprocess images
    imgs_pre = segment.preprocess(
        imgs, median_filter=False,
        rescale_intensity_range=ui['rescale_range']
    )
    # Generate preprocessed viz
    fig, axes = view.vol_slices(
            imgs_pre,
            slices=ui['slices'],
            nslices=ui['nslices'],
            print_slices=False,
            fig_w=7.5,
            dpi=300
        )
    fig_n += 1
    segment.output_checkpoints(
        fig, show=show_checkpoints, save_path=checkpoint_save_dir,
        fn_n=fig_n, fn_suffix='preprocessed-imgs')
    # Generate preprocessed imgs viz
    fig, axes = view.vol_slices(
            imgs_pre,
            slices=ui['slices'],
            nslices=ui['nslices'],
            print_slices=False,
            fig_w=7.5,
            dpi=300
        )
    fig_n += 1
    segment.output_checkpoints(
        fig, show=show_checkpoints, save_path=checkpoint_save_dir,
        fn_n=fig_n, fn_suffix='preprocessed-imgs')

    #-----------------------#
    # Semantic segmentation #
    #-----------------------#
    print()
    # Calc semantic seg threshold values and generate histogram
    thresholds, fig, ax = segment.threshold_multi_min(
        imgs_pre, nbins=ui['thresh_nbins'], return_fig_ax=True,
        ylims=ui['thresh_hist_ylims']
    )
    fig_n += 1
    segment.output_checkpoints(
        fig, show=show_checkpoints, save_path=checkpoint_save_dir,
        fn_n=fig_n, fn_suffix='semantic-seg-hist')
    # Segment images with threshold values
    imgs_semantic = segment.isolate_classes(imgs_pre, thresholds)
    # Calc particle to binder ratio (voxels)
    particles_to_binder = segment.calc_voxel_stats(imgs_semantic)
    # Generate semantic label viz
    fig, axes = view.vol_slices(
            imgs_semantic,
            slices=ui['slices'],
            nslices=ui['nslices'],
            print_slices=False,
            fig_w=7.5,
            dpi=300
        )
    fig_n += 1
    segment.output_checkpoints(
        fig, show=show_checkpoints, save_path=checkpoint_save_dir,
        fn_n=fig_n, fn_suffix='semantic-seg-imgs')

    #-----------------------#
    # Instance segmentation #
    #-----------------------#
    if ui['perform_seg']:
        print()
        # Clear up memory
        imgs = None
        imgs_med = None
        imgs_pre = None
        imgs_instance = segment.watershed_segment(
            imgs_semantic==2,
            min_peak_distance=ui['min_peak_dist'],
            exclude_borders=ui['exclude_borders'],
            return_dict=False
        )
        # Generate instance label viz
        fig, axes = view.color_labels(
            imgs_instance,
            slices=ui['slices'],
            nslices=ui['nslices'],
            fig_w=7.5,
            dpi=300
        )
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='instance-seg-imgs')
        # Merge semantic and instance segs to represent binder and particles
        imgs_labeled = segment.merge_segmentations(imgs_semantic, imgs_instance)

    #-------------#
    # Save voxels #
    #-------------#
    if ui['save_voxels']:
        if['perform_seg']:
            segment.save_images(
                imgs_labeled,
                Path(ui['out_dir_path']) / f"{ui['out_prefix']}_labeled_voxels"
            )
        else:
            segment.save_images(
                imgs_semantic,
                Path(ui['out_dir_path']) / f"{ui['out_prefix']}_semantic_voxels"
            )

    #----------------------------------------#
    # Create Surface Meshes of Each Particle #
    #----------------------------------------#
    if ui['perform_seg'] and ui['save_stls']:
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
            voxel_step_size=ui['step_size'],
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

