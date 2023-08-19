import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow segments an image stack in which all particles are labeled'
    ' with the integer 2. Outputs can be instance-labeled TIF stack and/or STL'
    ' files. Developed for Segmentflow v0.0.3.'
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
    'B. Segmentation' : {
        'min_peak_distance' : '01. Minimum distance between particle centers',
    },
    'C. Output' : {
        'out_dir_path'     : '01. Path to save output dir',
        'overwrite'        : '02. Overwrite files',
        'out_prefix'       : '03. Output prefix',
        'nslices'          : '04. Number of slices in checkpoint plots',
        'save_checkpoints' : '05. Save checkpoint figures',
        'save_stl'         : '06. Save STL file',
        'save_voxels'      : '07. Save voxel TIF stack'
    },
}

DEFAULT_VALUES = {
    'in_dir_path'       : 'REQUIRED',
    'file_suffix'       : '.tif',
    'slice_crop'        : None,
    'row_crop'          : None,
    'col_crop'          : None,
    'spatial_res'       : 1,
    'min_peak_distance' : 10,
    'out_dir_path'      : 'REQUIRED',
    'out_prefix'        : '',
    'overwrite'         : True,
    'nslices'           : 5,
    'save_checkpoints'  : True,
    'save_stl'          : True,
    'save_voxels'       : True,
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
    imgs_semantic = segment.load_images(
        ui['in_dir_path'],
        slice_crop=ui['slice_crop'],
        row_crop=ui['row_crop'],
        col_crop=ui['col_crop'],
        convert_to_float=True,
        file_suffix=ui['file_suffix']
    )
    fig, axes = view.plot_slices(
        imgs_semantic,
        nslices=ui['nslices'],
        fig_w=7.5,
        dpi=300
    )
    if ui['save_checkpoints'] == 'show':
        plt.show()
    elif ui['save_checkpoints'] == True:
        plt.savefig(
            Path(ui['out_dir_path'])
            / f'{str(fig_n).zfill(n_fig_digits)}-semantic-seg-imgs.png')

    #-------------------#
    # Segment particles #
    #-------------------#
    print()
    imgs_labeled = segment.watershed_segment(
        imgs_semantic==2,
        min_peak_distance=ui['min_peak_distance'],
        exclude_borders=False,
        return_dict=False
    )
    fig, axes = view.plot_color_labels(
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

    #--------------#
    # Save outputs #
    #--------------#
    # Save STLs
    print()
    if ui['save_stl']:
        segment.save_as_stl_files(
            imgs_labeled,
            ui['out_dir_path'],
            ui['out_prefix'],
            make_new_save_dir=True,
            spatial_res=ui['spatial_res'],
            stl_overwrite=ui['overwrite']
        )
    # Save images
    if ui['save_voxels']:
        segment.save_images(
            imgs_labeled,
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

