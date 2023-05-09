import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view, mesh
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow segments spheres from a simulated CT scan and'
    ' outputs a labeled TIF stack and/or STL files corresponding'
    ' to each segmented particle. Developed for v0.0.1.'
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
        'view_semantic' : 'View semantic images',
        'view_labeled'  : 'View labeled images',
    },
    'Segmentation' : {
        'perform_seg'       : 'Perform instance segmentation',
        'min_peak_dist'     : 'Min peak distance',
        'exclude_borders'   : 'Exclude border particles',
        'save_voxels'       : 'Save instance labeled images',
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
    'spatial_res'          : 1,
    'out_dir_path'         : 'REQUIRED',
    'out_prefix'           : '',
    'overwrite'            : True,
    'view_slices'          : [75, 125, 175],
    'view_raw'             : True,
    'view_semantic'        : True,
    'view_labeled'         : True,
    'perform_seg'          : True,
    'min_peak_dist'        : 6,
    'exclude_borders'      : True,
    'save_voxels'          : False,
    'create_stls'          : True,
    'n_erosions'           : 0,
    'post_seg_med_filter'  : False,
    'voxel_step_size'      : 1,
    'suppress_save_msg'    : True,
    'mesh_smooth_n_iters'  : 1,
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

    #-----------------------#
    # Semantic segmentation #
    #-----------------------#
    print()
    imgs_semantic = imgs.copy()
    imgs_semantic[imgs < imgs.max()] = 0
    if ui['view_semantic']:
        fig, axes = view.plot_slices(
                imgs_semantic,
                slices=ui['view_slices'],
                print_slices=False,
                fig_w=7.5,
                dpi=100
            )

    #----------------#
    # Segment images #
    #----------------#
    if ui['perform_seg']:
        print()
        imgs = None
        imgs_labeled = segment.watershed_segment(
            imgs_semantic,
            min_peak_distance=ui['min_peak_dist'],
            exclude_borders=ui['exclude_borders'],
            return_dict=False
        )
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
                ui['out_dir_path'],
                smooth_iter=ui['mesh_smooth_n_iters'],
                simplify_n_tris=ui['mesh_simplify_n_tris'],
                iterative_simplify_factor=ui['mesh_simplify_factor'],
                recursive_simplify=False, resave_mesh=True
            )

    #-------------------------#
    # Plot figures if enabled #
    #-------------------------#
    if (
        ui['view_raw'] or ui['view_semantic'] or ui['view_labeled']
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

