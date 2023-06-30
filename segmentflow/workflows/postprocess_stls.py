import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view, mesh
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow loads images with labeled binder and particles and outputs'
    ' STL files corresponding to each segmented particle (skipping label 1'
    ' assumed to be binder).'
    ' Developed for v0.0.3 but will most likely work for other versions.'
)

CATEGORIZED_INPUT_SHORTHANDS = {
    'Files' : {
        'in_dir_path'  : 'Input dir path',
        'file_suffix'  : 'File suffix',
        'slice_crop'   : 'Slice crop',
        'row_crop'     : 'Row crop',
        'col_crop'     : 'Column crop',
        'spatial_res'  : 'Pixel size',
        'out_dir_path' : 'Output dir path',
        'out_prefix'   : 'Output prefix',
        'overwrite'    : 'Overwrite files'
    },
    'STL' : {
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
    'suppress_save_msg'    : True,
    'n_erosions'           : 0,
    'post_seg_med_filter'  : False,
    'spatial_res'          : 1,
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
    imgs_labeled = segment.load_images(
        ui['in_dir_path'],
        slice_crop=ui['slice_crop'],
        row_crop=ui['row_crop'],
        col_crop=ui['col_crop'],
        file_suffix=ui['file_suffix']
    )

    #-------------#
    # Voxel stats #
    #-------------#
    print()
    segment.calc_voxel_stats(imgs_labeled)

    #----------------------------------------#
    # Create Surface Meshes of Each Particle #
    #----------------------------------------#
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
            recursive_simplify=False,
            resave_mesh=True
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

