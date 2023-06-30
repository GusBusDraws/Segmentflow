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
    '1 Files' : {
        'stl_dir_path' : '00 Input STL dir path',
        'skip_first_stl'   : '01 Skip first STL',
        'out_dir_path' : '02 Output dir path',
        'out_prefix'   : '03 Output prefix',
        'overwrite'    : '04 Overwrite files'
    },
    '2 STL' : {
        'suppress_save_msg'    : '00 Suppress save message for each STL file',
        'mesh_smooth_n_iters'  : '01 Number of smoothing iterations',
        'mesh_simplify_n_tris' : '02 Target number of triangles/faces',
        'mesh_simplify_factor' : '03 Simplification factor per iteration',
    },
}

DEFAULT_VALUES = {
    'stl_dir_path'         : 'REQUIRED',
    'skip_first_stl'       : True,
    'out_dir_path'         : 'REQUIRED',
    'out_prefix'           : '',
    'overwrite'            : False,
    'suppress_save_msg'    : True,
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
    #----------------------------------------------#
    # Postprocess surface meshes for each particle #
    #----------------------------------------------#
    print()
    out_dir_name = Path(ui['stl_dir_path']).stem
    if ui['mesh_smooth_n_iters'] >= 1:
        out_dir_name = out_dir_name + '_smooth'
    if ui['mesh_simplify_n_tris'] is not None:
        out_dir_name = out_dir_name + f"_{ui['mesh_simplify_n_tris']}-tris"
    out_dir = Path(ui['out_dir_path']) / out_dir_name
    out_dir_path = str(out_dir.resolve())
    # Iterate through each STL file, load the mesh, and smooth/simplify
    mesh.postprocess_meshes(
        ui['stl_dir_path'],
        skip_first_stl=ui['skip_first_stl'],
        smooth_iter=ui['mesh_smooth_n_iters'],
        simplify_n_tris=ui['mesh_simplify_n_tris'],
        iterative_simplify_factor=ui['mesh_simplify_factor'],
        recursive_simplify=False,
        resave_mesh=False,
        save_dir_path=out_dir_path
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

