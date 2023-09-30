import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from segmentflow import segment, view
from skimage import segmentation
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow takes a stack of already-segmented, instance-labeled images'
    ' and creates a directory of STL files. Developed for Segmentflow v0.0.3.'
)

CATEGORIZED_INPUT_SHORTHANDS = {
    'A. Input' : {
        'in_dir_path'     : '01. Input dir path',
        'file_suffix'     : '02. File suffix',
        'slice_crop'      : '03. Slice crop',
        'row_crop'        : '04. Row crop',
        'col_crop'        : '05. Column crop',
        'spatial_res'     : '06. Pixel size',
        'exclude_borders' : '07. Exclude border particles',
    },
    'B. Output' : {
        'out_dir_path'     : '01. Path to save output dir',
        'overwrite'        : '02. Overwrite files',
        'out_prefix'       : '03. Output prefix',
        'nslices'          : '04. Number of slices in checkpoint plots',
        'slices'           : '05. Specify slices for checkpoint plots',
        'save_checkpoints' : '06. Save checkpoint figures',
        'save_stls'        : '07. Save STL files',
    },
}

DEFAULT_VALUES = {
    'in_dir_path'      : 'REQUIRED',
    'file_suffix'      : '.tif',
    'slice_crop'       : None,
    'row_crop'         : None,
    'col_crop'         : None,
    'spatial_res'      : 1,
    'out_dir_path'     : 'REQUIRED',
    'out_prefix'       : '',
    'overwrite'        : True,
    'nslices'          : 4,
    'slices'           : None,
    'save_checkpoints' : True,
    'save_stls'        : True,
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
    checkpoints_save_dir = ui['out_dir_path']
    #-------------#
    # Load images #
    #-------------#
    print()
    labels = segment.load_images(
        ui['in_dir_path'],
        slice_crop=ui['slice_crop'],
        row_crop=ui['row_crop'],
        col_crop=ui['col_crop'],
        file_suffix=ui['file_suffix']
    )
    n_fig_digits = 2
    fig_n = 0
    fig, axes = view.slices(
        labels,
        slices=ui['slices'],
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=300
    )
    segment.output_checkpoints(
        fig, show=show_checkpoints, save_path=checkpoints_save_dir,
        fn_n=fig_n, fn_suffix='label-imgs.png')

    #-----------------------------#
    # Exclude particles on border #
    #-----------------------------#
    if ui['exclude_borders']:
        print()
        print('Excluding border particles...')
        print('--> Counting particles...')
        # Subtract 1 to account for background label
        n_particles = len(np.unique(labels)) - 1
        print(
            '--> Number of particles before border exclusion: ',
            str(n_particles))
        labels = segmentation.clear_border(labels)
        print('--> Counting particles...')
        # Subtract 1 to account for background label
        n_particles = len(np.unique(labels)) - 1
        print(
            '--> Number of particles after border exclusion: ',
            str(n_particles))

    #-----------------------------#
    # Exclude particles on border #
    #-----------------------------#
    fig, axes = view.color_labels(
        labels,
        slices=ui['slices'],
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=300
    )
    fig_n += 1
    segment.output_checkpoints(
        fig, show=show_checkpoints, save_path=checkpoints_save_dir,
        fn_n=fig_n, fn_suffix='color-label-imgs.png')


    #--------------#
    # Save outputs #
    #--------------#
    if ui['save_stls']:
        # Save STLs
        print()
        segment.save_as_stl_files(
            labels,
            ui['out_dir_path'],
            ui['out_prefix'],
            make_new_save_dir=True,
            spatial_res=ui['spatial_res'],
            stl_overwrite=ui['overwrite']
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

