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
    'Files' : {
        'in_dir_path'  : 'Directory of labeled particle images',
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
    'in_dir_path'  : 'REQUIRED',
    'out_dir_path' : 'REQUIRED',
    'out_prefix'   : '',
    'file_suffix'  : '.tif',
    'slice_crop'   : None,
    'row_crop'     : None,
    'col_crop'     : None,
}

def workflow(argv):
    #----------------------#
    # Read YAML input file #
    #----------------------#
    # ui = segment.process_args(argv, WORKFLOW_NAME, WORKFLOW_DESCRIPTION)
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
        'min_peak_dist' : 6,
        'exclude_borders' : True,
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
