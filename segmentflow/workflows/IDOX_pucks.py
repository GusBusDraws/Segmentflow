import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view, mesh
from skimage import filters, util
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow segments IDOX particles from a binder in CT scans of'
    ' of pressed pucks taken with the CSM Zeiss Versa x-ray source.'
    ' Manual thresholds are set for semantic segmentation.'
    ' The output can be labeled voxels following semantic segmentation or'
    ' instance segmentation, and/or STL files corresponding to each'
    ' instance-segmented particle.'
    ' Developed for Segmentflow v0.0.3.'
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
    'B. Output' : {
        'out_dir_path'      : '01. Path to save output dir',
        'overwrite'         : '02. Overwrite files',
        'out_prefix'        : '03. Output prefix',
        'slices'            : '04. Specify slices to plot',
        'nslices'           : '05. Number of slices in checkpoint plots',
        'save_stls'         : '06. Save STL files',
        'suppress_save_msg' : '07. Suppress save message for each STL file',
        'save_voxels'       : '08. Save voxel TIF stack'
    },
    'C. Preprocessing' : {
        'pre_seg_med_filter' : '01. Apply median filter',
        'rescale_range'      : '02. Range for rescaling intensity (percentile)',
    },
    'D. Segmentation' : {
        'thresh_vals'       : '01. Threshold values for semantic segmentation',
        'thresh_hist_ylims' : '02. Upper and lower y-limits of histogram',
        'fill_holes'        : '03. Fill holes in semantic segmentation',
        'perform_seg'       : '04. Perform instance segmentation',
        'min_peak_dist'     : '05. Min distance between region centers (pixels)',
        'exclude_borders'   : '06. Exclude border particles',
    },
    'E. Surface Meshing' : {
        'n_erosions'           : '01. Number of pre-surface meshing erosions',
        'post_seg_med_filter'  : '02. Smooth voxels with median filtering',
        'voxel_step_size'      : '03. Marching cubes voxel step size',
        'mesh_smooth_n_iters'  : '04. Number of smoothing iterations',
        'mesh_simplify_n_tris' : '05. Target number of triangles/faces',
        'mesh_simplify_factor' : '06. Simplification factor per iteration',
    },
}

DEFAULT_VALUES = {
    'in_dir_path'          : 'REQUIRED',
    'file_suffix'          : 'tiff',
    'slice_crop'           : None,
    'row_crop'             : None,
    'col_crop'             : None,
    'spatial_res'          : 1,
    'out_dir_path'         : 'REQUIRED',
    'out_prefix'           : '',
    'overwrite'            : False,
    'nslices'              : 4,
    'slices'               : None,
    'save_stls'            : True,
    'suppress_save_msg'    : True,
    'save_voxels'          : False,
    'pre_seg_med_filter'   : False,
    'rescale_range'        : None,
    'thresh_vals'          : [30000, 62000],
    'thresh_hist_ylims'    : [0, 2e7],
    'fill_holes'           : False,
    'perform_seg'          : True,
    'min_peak_dist'        : 6,
    'exclude_borders'      : False,
    'n_erosions'           : 0,
    'post_seg_med_filter'  : False,
    'voxel_step_size'      : 1,
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
        DEFAULT_VALUES)
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
    print(f'{imgs_pre.dtype=}')
    imgs_pre = util.img_as_uint(imgs_pre)
    print(f'{imgs_pre.dtype=}')
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

    #-----------------------#
    # Semantic segmentation #
    #-----------------------#
    print()
    # Sort threshold values in descending order
    # thresholds = sorted(ui['thresh_vals'], reverse=True)
    thresholds = sorted(ui['thresh_vals'])
    # Calc semantic seg threshold values and generate histogram
    fig, ax = view.histogram(imgs_pre, mark_values=thresholds)
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

    #------------------#
    # Fill small holes #
    #------------------#
    if ui['fill_holes'] is not None:
        print()
        imgs_semantic = segment.fill_holes(imgs_semantic)
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
            fn_n=fig_n, fn_suffix='semantic-seg-imgs-holes-filled')

    #----------------#
    # Segment images #
    #----------------#
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
        # Merge semantic and instance segs to represent binder and particles
        imgs_labeled = segment.merge_segmentations(imgs_semantic, imgs_instance)
        # Post-seg median filter
        if ui['post_seg_med_filter']:
            imgs_labeled = filters.median(imgs_labeled)
            imgs_instance[imgs_labeled == 1] = 0
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
        # Clear up memory
        imgs_semantic = None
        imgs_instance = None
        # Remove voxels corresponding to binder to save only particles as STLs
        imgs_labeled[imgs_labeled == 1] = 0
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
            fn_n=fig_n, fn_suffix='watertight-chart')

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
                Path(ui['out_dir_path']) / f"{ui['out_prefix']}_STLs",
                smooth_iter=ui['mesh_smooth_n_iters'],
                simplify_n_tris=ui['mesh_simplify_n_tris'],
                iterative_simplify_factor=ui['mesh_simplify_factor'],
                recursive_simplify=False,
                resave_mesh=False,
                save_dir_path=(
                    Path(ui['out_dir_path'])
                    / f"{ui['out_prefix']}_processed-STLs",
                )
            )


if __name__ == '__main__':
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    print(f'Beginning Workflow: {WORKFLOW_NAME}')
    print()
    workflow(sys.argv[1:])
    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Successful Completion')
    print('~~~~~~~~~~~~~~~~~~~~~')
    print()

