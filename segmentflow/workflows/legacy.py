import getopt
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import scipy
import scipy.ndimage as ndi
from skimage import (
        exposure, feature, filters, morphology, measure,
        segmentation, util
)
from stl import mesh
import sys
# Local imports
import segment
import view
import mesh
import yaml


WORKFLOW_NAME = Path(__name__).stem

#~~~~~~~~~~~#
# Utilities #
#~~~~~~~~~~~#
def fatalError(message):
    print()
    print('---------------------------')
    print()
    print('A fatal error has occurred.')
    print()
    print('---------------------------')
    print()
    print(message)
    print()

def help():
    print()
    print('----------------------------------------------------------------')
    print()
    print(f'This is {WORKFLOW_NAME}.py, a workflow script for Segmentflow.')
    print()
    print(
        'This workflow is the  to segment particles in a CT scan'
        ' according to the preferences set in an input YAML file.'
        ' Output can be a labeled TIF stack and/or STL files corresponding'
        ' to each segmented particle.'
    )
    print()
    print('----------------------------------------------------------------')
    print()
    print('Usage:')
    print()
    print(f'python segmentflow.workflows.{WORKFLOW_NAME}.py -i path/to/input_file.yml')
    print()
    print(
        'where input_file.yml is the path to the YAML input file.'
        ' See the example input file in the repo top-level directory'
        ' to learn more about the content (inputs) of the input file.'
    )
    print()

def load_inputs(yaml_path):
    """Load input file and output a dictionary filled with default values
    for any inputs left blank.
    ----------
    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to input YAML file.
    -------
    Returns
    -------
    dict
        Dict containing inputs stored according to shorthands.
    ------
    Raises
    ------
    ValueError
        Raised if "CT Scan Dir" or "STL Dir" inputs left blank.
    """
    # Open YAML file and read inputs
    stream = open(yaml_path, 'r')
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)   # User Input
    stream.close()
    # Create shorthand dictionary for inputs in yaml_dict
    categorized_input_shorthands = {
        'Files' : {
            'ct_img_dir'          : 'CT Scan Dir',
            'stl_dir_location'    : 'STL Dir',
            'output_fn_base'      : 'STL Prefix',
            'stl_overwrite'       : 'Overwrite Existing STL Files',
            'single_particle_iso' : 'Particle ID',
            'suppress_save_msg'   : 'Suppress Save Messages',
        },
        'Load' : {
            'file_suffix' : 'File Suffix',
            'slice_crop'  : 'Slice Crop',
            'row_crop'    : 'Row Crop',
            'col_crop'    : 'Col Crop',
        },
        'Preprocess' : {
            'pre_seg_med_filter' : 'Apply Median Filter',
            'rescale_range'      : 'Rescale Intensity Range',
        },
        'Binarize' : {
            'n_otsu_classes'     : 'Number of Otsu Classes',
            'downsample_factor'  : 'Image Stack Downsample Factor',
            'n_selected_classes' : 'Number of Classes to Select',
            'save_classes'       : 'Save Isolated Classes',
        },
        'Segment' : {
            'perform_seg'      : 'Perform Segmentation',
            'use_int_dist_map' : 'Use Integer Distance Map',
            'min_peak_dist'    : 'Min Peak Distance',
            'exclude_borders'  : 'Exclude Border Particles',
        },
        'STL' : {
            'create_stls'          : 'Create STL Files',
            'n_erosions'           : 'Number of Pre-Surface Meshing Erosions',
            'post_seg_med_filter'  : 'Smooth Voxels with Median Filtering',
            'spatial_res'          : 'Pixel-to-Length Ratio',
            'voxel_step_size'      : 'Marching Cubes Voxel Step Size',
            'mesh_smooth_n_iters'  : 'Number of Smoothing Iterations',
            'mesh_simplify_n_tris' : 'Target number of Triangles/Faces',
            'mesh_simplify_factor' : 'Simplification factor Per Iteration',
        },
        'Plot' : {
            'seg_fig_show'     : 'Segmentation Plot Create Figure',
            'seg_fig_n_imgs'   : 'Segmentation Plot Number of Images',
            'seg_fig_slices'   : 'Segmentation Plot Slices',
            'seg_fig_plot_max' : 'Segmentation Plot Show Maxima',
            'label_fig_show'   : 'Particle Labels Plot Create Figure',
            'label_fig_idx'    : 'Particle Labels Plot Image Index',
            'stl_fig_show'     : 'STL Plot Create Figure',
        }
    }
    # Dict of default values to replace missing or blank input entries
    default_values = {
        'ct_img_dir'           : 'REQUIRED',
        'stl_dir_location'     : 'REQUIRED',
        'output_fn_base'       : '',
        'stl_overwrite'        : False,
        'single_particle_iso'  : None,
        'suppress_save_msg'    : False,
        'file_suffix'          : 'tiff',
        'slice_crop'           : None,
        'row_crop'             : None,
        'col_crop'             : None,
        'pre_seg_med_filter'   : False,
        'rescale_range'        : None,
        'n_otsu_classes'       : 3,
        'downsample_factor'    : 1,
        'n_selected_classes'   : 1,
        'save_classes'         : False,
        'perform_seg'          : True,
        'use_int_dist_map'     : False,
        'min_peak_dist'        : 7,
        'exclude_borders'      : False,
        'create_stls'          : True,
        'n_erosions'           : 0,
        'post_seg_med_filter'  : False,
        'spatial_res'          : 1,
        'voxel_step_size'      : 1,
        'mesh_smooth_n_iters'  : None,
        'mesh_simplify_n_tris' : None,
        'mesh_simplify_factor' : None,
        'seg_fig_show'         : False,
        'seg_fig_n_imgs'       : 3,
        'seg_fig_slices'       : None,
        'seg_fig_plot_max'     : False,
        'label_fig_show'       : False,
        'label_fig_idx'        : 0,
        'stl_fig_show'         : False,
    }
    # Iterate through input shorthands to make new dict with shorthand keys (ui)
    ui = {}
    for category, input_shorthands in categorized_input_shorthands.items():
        for shorthand, input in input_shorthands.items():
            # try-except to make sure each input exists in input file
            try:
                ui[shorthand] = yaml_dict[category][input]
            except KeyError as error:
                # Set missing inputs to None
                ui[shorthand] = None
            finally:
                # For any input that is None (missing or left blank),
                # Change None value to default value from default_values dict
                if ui[shorthand] == None:
                    # Raise ValueError if default value is listed as 'REQUIRED'
                    if default_values[shorthand] == 'REQUIRED':
                        raise ValueError(
                            f'Must provide value for "{input}"'
                            f' in input YAML file.'
                        )
                    else:
                        # Set default value as denoted in default_values.
                        # Value needs to be set in yaml_dict to be saved in the
                        # copy of the insput, but also in the ui dict to be
                        # used in the code
                        yaml_dict[category][input] = default_values[shorthand]
                        ui[shorthand] = default_values[shorthand]
                        if default_values[shorthand] is not None:
                            print(
                                f'Value for "{input}" not provided.'
                                f' Setting to default value:'
                                f' {default_values[shorthand]}'
                            )
    stl_dir = Path(ui['stl_dir_location'])
    if not stl_dir.is_dir():
        stl_dir.mkdir()
    # Copy YAML input file to output dir
    with open(
        Path(ui['stl_dir_location'])
        / f'{ui["output_fn_base"]}input.yml', 'w'
    ) as file:
        output_yaml = yaml.dump(yaml_dict, file)
    return ui

#~~~~~~~~~~#
# Workflow #
#~~~~~~~~~~#
def workflow(argv):
    #----------------------------#
    # Get command-line arguments #
    #----------------------------#
    try:
        opts, args = getopt.getopt(argv,"hf:",["ifile=","ofile="])
    except getopt.GetoptError:
        fatalError(
            f'Error in command-line arguments.'
            f' Enter "python -m segmentflow.workflow.{WORKFLOW_NAME} -h"'
            f' for more help'
        )
    yaml_file = ''
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        if opt == "-i":
            yaml_file = str(arg)

    #----------------------#
    # Read YAML input file #
    #----------------------#
    if yaml_file == '':
        fatalError(
            f'No input file specified.'
            f' Enter "python -m segmentflow.workflow.{WORKFLOW_NAME} -h"'
            f' for more help'
        )
    else:
        # Load YAML inputs into a dictionary
        ui = segment.load_inputs(yaml_file)

    #-------------#
    # Load images #
    #-------------#
    print()
    imgs = segment.load_images(
        ui['ct_img_dir'],
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

    #-----------------#
    # Binarize images #
    #-----------------#
    print()
    imgs_binarized, thresh_vals = segment.binarize_multiotsu(
        imgs_pre, n_otsu_classes=ui['n_otsu_classes'],
        n_selected_thresholds=ui['n_selected_classes'],
    )
    if ui['save_classes']:
        segment.save_isolated_classes(
            imgs_pre, thresh_vals, ui['stl_dir_location']
        )

    #----------------#
    # Segment images #
    #----------------#
    if ui['perform_seg']:
        print()
        segment_dict = segment.watershed_segment(
            imgs_binarized, min_peak_distance=ui['min_peak_dist'],
            use_int_dist_map=ui['use_int_dist_map'],
            exclude_borders=ui['exclude_borders'], return_dict=True
        )

    #----------------------------------------#
    # Create Surface Meshes of Each Particle #
    #----------------------------------------#
    if ui['create_stls']:
        if ui['perform_seg']:
            voxels_to_mesh = segment_dict['integer-labels']
        else:
            voxels_to_mesh = imgs_binarized
        print()
        segment.save_as_stl_files(
            voxels_to_mesh,
            ui['stl_dir_location'],
            ui['output_fn_base'],
            suppress_save_msg=ui['suppress_save_msg'],
            slice_crop=ui['slice_crop'],
            row_crop=ui['row_crop'],
            col_crop=ui['col_crop'],
            stl_overwrite=ui['stl_overwrite'],
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
    if ui['seg_fig_show']:
        fig_seg_steps, axes_seg_steps = view.plot_segment_steps(
            imgs, imgs_pre, imgs_binarized, segment_dict,
            n_imgs=ui['seg_fig_n_imgs'], slices=ui['seg_fig_slices'],
            plot_maxima=ui['seg_fig_plot_max']
        )
    if ui['label_fig_show']:
        fig_labels, ax_labels = view.plot_particle_labels(
                segment_dict, ui['label_fig_idx'])
    if ui['stl_fig_show']:
        fig_stl, ax_stl = view.plot_stl(ui['stl_dir_location'])
    if ui['seg_fig_show'] or ui['label_fig_show'] or ui['stl_fig_show']:
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

