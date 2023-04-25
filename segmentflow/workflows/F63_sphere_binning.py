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
import sys
import yaml
# Local imports
from segmentflow import segment
from segmentflow import view


WORKFLOW_DESCRIPTION = (
    'This workflow is the  to segment particles in a CT scan'
    ' according to the preferences set in an input YAML file.'
    ' Output can be a labeled TIF stack and/or STL files corresponding'
    ' to each segmented particle.'
)

WORKFLOW_NAME = Path(__file__).stem

CATEGORIZED_INPUT_SHORTHANDS = {
    'Files' : {
        'in_dir_path'  : 'Input dir',
        'out_dir_path' : 'Output dir path',
        'out_prefix'   : 'Output prefix',
        'file_suffix'  : 'File Suffix',
    },
    'Load' : {
        'slice_crop' : 'Slice Crop',
        'row_crop'   : 'Row Crop',
        'col_crop'   : 'Column Crop',
    },
}

DEFAULT_VALUES = {
    'in_dir'           : 'REQUIRED',
}

def help():
    print()
    print('----------------------------------------------------------------')
    print()
    print(f'This is {WORKFLOW_NAME}.py, a workflow script for Segmentflow.')
    print()
    print(WORKFLOW_DESCRIPTION)
    print()
    print('----------------------------------------------------------------')
    print()
    print('Usage:')
    print()
    print(
        f'python segmentflow.workflows.{WORKFLOW_NAME}.py'
        '-i path/to/input_file.yml'
    )
    print()
    print(
        'where input_file.yml is the YAML input file.'
        ' See the example input file at the top-level directory of the repo'
        ' to learn more about the content (inputs) of the input file.'
    )
    print()

def process_args(argv, workflow_name):
    # Get command-line arguments
    try:
        opts, args = getopt.getopt(argv, 'hf:', ['ifile=','ofile='])
    except getopt.GetoptError:
        print(
            'Error in command-line arguments.',
            'Enter "python -m segmentflow.workflow.{WORKFLOW_NAME} -h"'
            ' for more help',
            sep='\n'
        )
    yaml_file = ''
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        if opt == "-i":
            yaml_file = str(arg)
    if yaml_file == '':
        raise ValueError(
            f'No input file specified.',
            f'Enter "python -m segmentflow.workflow.{WORKFLOW_NAME} -h"'
            f' for more help', sep='\n'
        )
    else:
        # Load YAML inputs into a dictionary
        ui = segment.load_inputs(argv)
        return ui

def workflow(argv):
    #----------------------#
    # Read YAML input file #
    #----------------------#
    # ui = process_args(argv)
    ui = {
        'in_dir_path' : 'c:/Users/cgusb/Research/mhe-analysis/data/F63tiff',
        'slice_crop' : [400, 550],
        'row_crop' : [400, 550],
        'col_crop' : [400, 550],
        'file_suffix' : 'tiff',
        'pre_seg_med_filter' : 3,
        'rescale_range' : [0, 99.9],
        'nbins_multi_min' : 100,
        'plot_thresholds' : True,
    }
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
        imgs,
        median_filter=ui['pre_seg_med_filter'],
        rescale_intensity_range=ui['rescale_range']
    )

    #-----------------#
    # Binarize images #
    #-----------------#
    print()
    thresholds = segment.threshold_multi_min(
        imgs_pre,
        nbins=ui['nbins_multi_min'],
    )
    if ui['plot_thresholds']:
        fig, ax = view.plot_thresholds(imgs_pre, thresholds)
    imgs_semantic = segment.isolate_classes(imgs_pre, thresholds)

    #----------------#
    # Segment images #
    #----------------#
    # print()
    # imgs_labeled = segment.watershed_segment(
    #     imgs_semantic == 2,
    #     min_peak_distance=ui['min_peak_dist'],
    #     exclude_borders=ui['exclude_borders'],
    #     return_dict=False
    # )

    #---------------------------#
    # Analyze size distribution #
    #---------------------------#
    plt.show()

if __name__ == '__main__':
    print()
    print(
        '~~~~~~~~~~~~~~~~~~~~~~~',
        'Welcome to Segmentflow!',
        '~~~~~~~~~~~~~~~~~~~~~~~',
        sep='\n'
    )
    print()
    print('Beginning workflow:', WORKFLOW_NAME, sep='\n')
    print()
    workflow(sys.argv[1:])
    print(
        '~~~~~~~~~~~~~~~~~~~~~~',
        'Successful Completion.',
        '~~~~~~~~~~~~~~~~~~~~~~',
        sep='\n'
    )
