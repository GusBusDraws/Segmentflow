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
# Local imports
from segmentflow import segment


WORKFLOW_DESCRIPTION = (
    'This workflow is the  to segment particles in a CT scan'
    ' according to the preferences set in an input YAML file.'
    ' Output can be a labeled TIF stack and/or STL files corresponding'
    ' to each segmented particle.'
)

WORKFLOW_NAME = Path(__file__).stem

CATEGORIZED_INPUT_SHORTHANDS = {
    'A. Input' : {
        'in_dir_path' : '01. Input dir path to labeled voxels',
        'file_suffix' : '02. File Suffix',
        'slice_crop'  : '03. Slice Crop',
        'row_crop'    : '04. Row Crop',
        'col_crop'    : '05. Column Crop',
    },
    'B: Output' : {
        'out_dir_path' : '01. Output dir path',
        'out_prefix'   : '02. Output prefix',
    },
}

DEFAULT_VALUES = {
    'in_dir_path'  : 'REQUIRED',
    'file_suffix'  : '.tif',
    'slice_crop'   : None,
    'row_crop'     : None,
    'col_crop'     : None,
    'out_dir_path' : 'REQUIRED',
    'out_prefix'   : '',
}

def workflow(argv):
    #----------------------#
    # Read YAML input file #
    #----------------------#
    # ui = segment.process_args(argv, WORKFLOW_NAME, WORKFLOW_DESCRIPTION)
    ui = {
        'in_dir_path' : 'c:/Users/gusb/Research/mhe-analysis/data/F63tiff',
        'slice_crop' : None,
        'row_crop' : None,
        'col_crop' : None,
        'file_suffix' : '.tif',
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
        convert_to_float=False,
        file_suffix=ui['file_suffix']
    )

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
