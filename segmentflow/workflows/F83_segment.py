import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view, mesh
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow segments F50 sand grains from a CT scan of a pressed puck'
    ' and outputs STL files corresponding to each segmented particle.'
    ' Developed for v0.0.3.'
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
        'nslices'           : '04. Number of slices in checkpoint plots',
        'save_checkpoints'  : '05. Save checkpoint figures',
        'save_stls'         : '06. Save STL file',
        'suppress_save_msg' : '07. Suppress save message for each STL file',
        'save_voxels'       : '08. Save voxel TIF stack'
    },
    'C. Preprocessing' : {
        'pre_seg_med_filter' : '01. Apply median filter',
        'rescale_range'      : '02. Range for rescaling intensity (percentile)',
    },
    'D. Segmentation' : {
        'thresh_nbins'      : '01. Histogram bins for calculating thresholds',
        'thresh_hist_ylims' : '02. Upper and lower y-limits of histogram',
        'perform_seg'       : '03. Perform instance segmentation',
        'min_peak_dist'     : '04. Min distance between region centers (pixels)',
        'exclude_borders'   : '05. Exclude border particles',
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
    'nslices'              : 5,
    'save_checkpoints'     : True,
    'save_stls'            : True,
    'suppress_save_msg'    : False,
    'save_voxels'          : False,
    'pre_seg_med_filter'   : False,
    'rescale_range'        : None,
    'thresh_nbins'         : 256,
    'view_thresh_hist'     : True,
    'thresh_hist_ylims'    : [0, 2e7],
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
    # Generate raw imgs viz
    fig, axes = view.slices(
            imgs,
            nslices=ui['nslices'],
            print_slices=False,
            fig_w=7.5,
            dpi=300
        )
    n_fig_digits = 2
    fig_n = 0
    if ui['save_checkpoints'] == 'show':
        plt.show()
    elif ui['save_checkpoints'] == True:
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-raw-imgs.png')

    #-------------------#
    # Preprocess images #
    #-------------------#
    print()
    imgs_pre = segment.preprocess(
        imgs, median_filter=ui['pre_seg_med_filter'],
        rescale_intensity_range=ui['rescale_range']
    )
    # Generate preprocessed viz
    fig, axes = view.slices(
            imgs_pre,
            nslices=ui['nslices'],
            print_slices=False,
            fig_w=7.5,
            dpi=300
        )
    fig_n += 1
    if ui['save_checkpoints'] == 'show':
        plt.show()
    elif ui['save_checkpoints'] == True:
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-preprocessed-imgs.png')

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
    if ui['save_checkpoints'] == 'show':
        plt.show()
    elif ui['save_checkpoints'] == True:
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-preprocessed-hist.png')
    # Segment images with threshold values
    imgs_semantic = segment.isolate_classes(imgs_pre, thresholds)
    # Calc particle to binder ratio (voxels)
    particle_to_binder = segment.calc_voxel_stats(imgs_semantic)
    # Generate semantic label viz
    fig, axes = view.slices(
            imgs_semantic,
            nslices=ui['nslices'],
            print_slices=False,
            fig_w=7.5,
            dpi=300
        )
    fig_n += 1
    if ui['save_checkpoints'] == 'show':
        plt.show()
    elif ui['save_checkpoints'] == True:
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-semantic-seg-imgs.png')

    #----------------#
    # Segment images #
    #----------------#
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
        # Generate instance label viz
        fig, axes = view.color_labels(
            imgs_labeled,
            nslices=ui['nslices'],
            fig_w=7.5,
            dpi=300
        )
        fig_n += 1
        if ui['save_checkpoints'] == 'show':
            plt.show()
        elif ui['save_checkpoints'] == True:
            plt.savefig(
                Path(ui['out_dir_path'])
                / f'{str(fig_n).zfill(n_fig_digits)}-instance-seg-imgs.png')

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
                Path(ui['out_dir_path']) / f"{ui['out_prefix']}_STLs",
                smooth_iter=ui['mesh_smooth_n_iters'],
                simplify_n_tris=ui['mesh_simplify_n_tris'],
                iterative_simplify_factor=ui['mesh_simplify_factor'],
                recursive_simplify=False,
                resave_mesh=True
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

