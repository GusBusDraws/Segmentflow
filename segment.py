#!/usr/bin/python3
#Requires: Python version >= 3.5
#          PyYAML (import yaml)

#~~~~~~~~~
# Packages
#~~~~~~~~~

from email.policy import default
from genericpath import isdir
import getopt
import imageio.v3 as iio
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import open3d as o3d
from pathlib import Path
import pandas as pd
from scipy import ndimage as ndi
import shutil
from skimage import ( color, exposure, feature, filters, 
    morphology, measure, segmentation, util )
from stl import mesh
import sys
import yaml

#~~~~~~~~~~
# Utilities
#~~~~~~~~~~

def fatalError(message):

    print()
    print('==')
    print('||')
    print('||   F A T A L   E R R O R')
    print('||')
    print("||  Sorry, there has been a fatal error.", end=" ")
    print(" Error message follows this banner.")
    print('||')
    print('==')
    print()
    print(message)
    print()
    exit(0)
    
def help():

    print()
    print('==')
    print('||')
    print('||   This is segment.py.')
    print('||')
    print("||  This script converts CT scans of samples containing", end=" ")
    print("particles to STL files, where")
    print('||  each STL file describes one of the particles in the sample.')
    print('||')
    print('==')
    print()
    print('Usage')
    print()
    print('   ./segment.py -f <inputFile.yml>')
    print()
    print("'where <inputFile.yml> is the path to your YAML input file.",end=" ")
    print("See the example input file")
    print("in the repo top-level directory to learn more about the", end=" ")
    print("content (inputs) of the input file.")
    print()
    exit(0)

#~~~~~~~~~~
# Functions
#~~~~~~~~~~

def load_inputs(yaml_path):
    """Load input file and output a dictionary filled with default values 
    for any inputs left blank.

    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to input YAML file.

    Returns
    -------
    dict
        Dict containing inputs stored according to shorthands.

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
            'rescale_range'         : 'Rescale Intensity Range',
        },
        'Binarize' : {
            'n_otsu_classes'     : 'Number of Otsu Classes',
            'n_selected_classes' : 'Number of Classes to Select',
        },
        'Segment' : {
            'use_int_dist_map' : 'Use Integer Distance Map',
            'min_peak_dist'    : 'Min Peak Distance',
            'exclude_borders'  : 'Exclude Border Particles',
        },
        'STL' : {
            'create_stls'            : 'Create STL Files',
            'n_erosions'             : 'Number of Pre-Surface Meshing Erosions',
            'post_seg_med_filter'    : 'Smooth Voxels with Median Filtering',
            'spatial_res'            : 'Pixel-to-Length Ratio',
            'voxel_step_size'        : 'Marching Cubes Voxel Step Size',
            'mesh_smooth_n_iters'    : 'Number of Smoothing Iterations',
            'mesh_simplify_n_tris'   : 'Target number of Triangles/Faces',
            'mesh_simplify_factor'   : 'Simplification factor Per Iteration',
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
        'n_otsu_classes'       : 2,
        'n_selected_classes'   : 1,
        'use_int_dist_map'     : False,
        'min_peak_dist'        : 1,
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
                                ' in input YAML file.')
                    else:
                        # Set default value as denoted in default_values. 
                        # Value needs to be set in yaml_dict to be saved in the 
                        # copy of the insput, but als in the ui dict to be used
                        # in the code
                        yaml_dict[category][input] = default_values[shorthand]
                        ui[shorthand] = default_values[shorthand]
                        if default_values[shorthand] is not None:
                            print(
                                    f'Value for "{input}" not provided. '    
                                    f'Setting to default value: '
                                    f'{default_values[shorthand]}')

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

def load_images(
    img_dir,
    slice_crop=None, 
    row_crop=None, 
    col_crop=None, 
    also_return_names=False,
    also_return_dir_name=False,
    convert_to_float=False,
    file_suffix='tiff',
    print_size=False,
):
    """Load images from path and return as list of 2D arrays. 
        Can also return names of images.

    Parameters
    ----------
    img_dir : str or Path
        Path to directory containing images to be loaded.
    slice_crop : list or None
        Cropping limits of slice dimension (imgs.shape[0]) of 3D array of 
        images. Essentially chooses subset of images from sorted image 
        directory. If None, all images/slices will be loaded. Defaults to None.
    row_crop : str or None
        Cropping limits of row dimension (imgs.shape[1]) of 3D array of images. 
        If None, all rows will be loaded. Defaults to None.
    col_crop : str or None
        Cropping limits of column dimension (imgs.shape[2]) of 3D array of 
        images. If None, all columns will be loaded. Defaults to None.
    also_return_names : bool, optional
        If True, returns a list of the names of the images in addition to the 
        list of images themselves. Defaults to False.
    also_return_dir_name : bool, optional
        If True, returns a string representing the name of the image directory 
        in addition to the list of images themselves. Defaults to False.
    convert_to_float : bool, optional
        If True, convert loaded images to floating point images, else retain 
        their original dtype. Defaults to False
    file_suffix : str, optional
        File suffix of images that will be loaded from img_dir. 
        Defaults to 'tif'
    print_size : bool, optional
        If True, print size of loaded images in GB. Defaults to False.

    Returns
    -------
    list, numpy.ndarray, or tuple
        List of arrays or 3D array representing images 
        (depending on return_3d_array), or if also_return_names is True, 
        list containing names of images from filenames is also returned.
    """
    print('Loading images...')
    img_dir = Path(img_dir)
    img_path_list = [
        path for path in img_dir.glob(f'*{file_suffix}')
    ]
    img_path_list.sort()
    if slice_crop is None:
        slice_crop = [0, len(img_path_list)]
    img_path_sublist = [
        img_path for i, img_path in enumerate(img_path_list) 
        if i in list(range(slice_crop[0], slice_crop[1]))
    ]
    n_slices = len(img_path_sublist)
    img = iio.imread(img_path_sublist[0])
    if row_crop is None:
        row_crop = [0, img.shape[0]]
    if col_crop is None:
        col_crop = [0, img.shape[1]]
    # Initialize 3D NumPy array to store loaded images
    imgs = np.zeros(
        (n_slices, row_crop[1] - row_crop[0], col_crop[1] - col_crop[0]),
        dtype=img.dtype
    )
    for i, img_path in enumerate(img_path_sublist):
        imgs[i, ...] = iio.imread(img_path)[row_crop[0]:row_crop[1], \
        col_crop[0]:col_crop[1]]
    print('--> Images loaded as 3D array: ', imgs.shape)
    if print_size:
        print('--> Size of array (GB): ', imgs.nbytes / 1E9)
    if also_return_names and also_return_dir_name:
        return imgs, [img_path.stem for img_path in img_path_list], img_dir.stem
    elif also_return_names:
        return imgs, [img_path.stem for img_path in img_path_list]
    elif also_return_dir_name:
        return imgs, img_dir.stem
    else:
        return imgs

def binarize_3d(
    imgs, 
    thresh_val=0.65, 
    fill_holes=64,
    return_process_dict=False
):
    """Creates binary images from list of images using a threshold value.

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing the floating point images to be binarized.
    thresh_val : float, optional
        Value to threshold point images. Defaults to 0.65 for floating 
        point images.
    fill_holes : str or int, optional
        If 'all', all holes will be filled, else if integer, all holes with an 
        area in pixels below that value will be filled in binary array/images. 
        Defaults to 64.
    return_process_dict : bool, optional
        If True, return a dictionary containing all processing steps instead 
        of last step only, defaults to False

    Returns
    -------
    numpy.ndarray or dict
        If return_process_dict is False, a 3D array representing the 
        hole-filled, binary images, else a dictionary is returned with a 
        3D array for each step in the binarization process.
    """
    smoothed = filters.gaussian(imgs)
    binarized = smoothed > thresh_val
    filled = binarized.copy()
    if fill_holes == 'all':
        for i in range((imgs.shape[0])):
            filled[i, :, :] = ndi.binary_fill_holes(binarized[i, :, :])
    else:
        filled = morphology.remove_small_holes(
            binarized, area_threshold=fill_holes
        )
    if return_process_dict:
        process_dict = {
            'binarized' : binarized,
            'holes-filled' : filled
        }
        return process_dict
    else:
        return filled

def preprocess(
    imgs,
    median_filter=False,
    rescale_intensity_range=None,
    print_size=False,
):
    """Preprocessing steps to perform on images.

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing slices of a 3D volume.
    rescale_intensity_range : None or 2-tuple, optional
        Intensity range (in percent) to clip intensity. Defaults to None.
    print_size : bool, optional
        If True, print the size of the preprocessed images in GB. 
        Defaults to False.

    Returns
    -------
    numpy.ndarray, list
        3D array of the shape imgs.shape containing binarized images; list of 
        threshold values used to create binarized images 
    """
    print('Preprocessing images...')
    imgs_pre = imgs.copy()
    # Apply median filter if median_filter is True
    if median_filter:
        print(f'--> Applying median filter...')
        imgs_pre = filters.median(imgs_pre)
    # Rescale intensity if intensity_range passed
    if rescale_intensity_range is not None:
        print(
                f'--> Rescaling intensities to percentile range '
                f'[{rescale_intensity_range[0]}, {rescale_intensity_range[1]}]'
                f'...')
        # Calculate low & high intensities
        rescale_low = np.percentile(imgs_pre, rescale_intensity_range[0])
        rescale_high = np.percentile(imgs_pre, rescale_intensity_range[1])
        # Clip low & high intensities
        imgs_pre = np.clip(imgs_pre, rescale_low, rescale_high)
        imgs_pre = exposure.rescale_intensity(
            imgs_pre, in_range='image', out_range='uint16'
        )
    print('--> Preprocessing complete')
    if print_size:
        print('--> Size of array (GB): ', imgs_pre.nbytes / 1E9)
    return imgs_pre

def binarize_multiotsu(
    imgs, 
    n_otsu_classes=2, 
    n_selected_thresholds=1, 
    exclude_borders=False,
    print_size=False,
):
    """Binarize stack of images (3D array) using multi-Otsu thresholding 
    algorithm.

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing slices of a 3D volume.
    n_otsu_classes : int, optional
        Number of classes to threshold images, by default 2
    n_selected_thresholds : int, optional
        Number of classes to group together (from the back of the thresholded 
        values array returned by multi-Otsu function) to create binary image, 
        by default 1
    exclude_borders : bool, optional
        If True, exclude particles that touch the border of the volume chunk 
        specified by slice/row/col crop in load_images(). Defaults to False.
    print_size : bool, optional
        If True, print size of binarized images in GB. Defaults to False.

    Returns
    -------
    numpy.ndarray, list
        3D array of the shape imgs.shape containing binarized images; list of 
        threshold values used to create binarized images 
    """
    print('Binarizing images...')
    imgs_binarized = np.zeros_like(imgs, dtype=np.uint8)
    print('--> Calculating Otsu threshold(s)...')
    imgs_flat = imgs.flatten()
    thresh_vals = filters.threshold_multiotsu(imgs_flat, n_otsu_classes)
    # In an 8-bit image (uint8), the max value is 255
    # The top regions are selected by counting backwards (-) with 
    # n_selected_thresholds
    imgs_binarized[imgs > thresh_vals[-n_selected_thresholds]] = 255
    # Remove regions of binary image at borders of array
    if exclude_borders:
        imgs_binarized = segmentation.clear_border(imgs_binarized)
    print('--> Binarization complete.')
    if print_size:
        print('--> Size of array (GB): ', imgs_binarized.nbytes / 1E9)
    return imgs_binarized, thresh_vals

def watershed_segment(
    imgs_binarized, 
    min_peak_distance=1,
    use_int_dist_map=False,
    exclude_borders=False,
    print_size=False,
    return_dict=False,
):
    """Create images with regions segmented and labeled using a watershed 
    segmentation algorithm.

    Parameters
    ----------
    binarized_imgs : numpy.ndarray
        3D DxMxN array representing D binary images with M rows and N columns 
        to be used in segmentation.
    min_peak_distance : int or str, optional
        Minimum distance (in pixels) of local maxima to be used to generate 
        seeds for watershed segmentation algorithm. 'median' can be passed to 
        use the radius of the circle with equivalent area to the median 
        binary region. Defaults to 1.
    use_int_dist_map : bool, optional
        If True, convert distance map to 16-bit array. Use with caution-- 
        changes segmentation results
    print_size : bool, optional
        If True, print the size of each item in the segmentation dictionary 
        in GB. Defautls to False.
    return_dict : bool, optional
        If true, return dict, else return 3D array with pixels labeled 
        corresponding to unique particle integers (see below)

    Returns
    -------
    if return_dict == True :
        dict
            Dictionary of 3D DxMxN arrays the segmentation steps and labeled 
            images. Keys for dict: 'binarized', 'distance-map', 
            'maxima-points', 'maxima-mask', 'seeds', 'integer-labels'
    if return_dict == False :
        numpy.ndarray
            3D DxMxN array representing segmented images with pixels labeled 
            corresponding to unique particle integers
    """
    print('Segmenting images...')
    dist_map = ndi.distance_transform_edt(imgs_binarized)
    if use_int_dist_map:
        dist_map = dist_map.astype(np.uint16)
    # If prompted, calculate equivalent median radius
    if min_peak_distance == 'median':
        regions = []
        for i in range(imgs_binarized.shape[0]):
            labels = measure.label(imgs_binarized[0, ...])
            regions += measure.regionprops(labels)
        areas = [region.area for region in regions]
        median_slice_area = np.median(areas)
        # Twice the radius of circle of equivalent area
        min_peak_distance = 2 * int(round(np.sqrt(median_slice_area) // np.pi))
        print(f'Calculated min_peak_distance: {min_peak_distance}')
    # Calculate the local maxima with min_peak_distance separation
    maxima = feature.peak_local_max(
        dist_map, 
        min_distance=min_peak_distance,
        exclude_border=False
    )
    # Assign a label to each point to use as seed for watershed seg
    maxima_mask = np.zeros_like(imgs_binarized, dtype=np.uint8)
    maxima_mask[tuple(maxima.T)] = 255
    seeds = measure.label(maxima_mask)
    # Release values to aid in garbage collection
    maxima_mask = None
    labels = segmentation.watershed(
        -1 * dist_map, seeds, mask=imgs_binarized
    )
    # Convert labels to smaller datatype is number of labels allows
    if np.max(labels) < 2**8:
        labels = labels.astype(np.uint8)
    elif np.max(labels) < 2**16:
        labels = labels.astype(np.uint16)
    # Release values to aid in garbage collection
    seeds = None
    # Count number of particles segmented
    n_particles = np.max(labels)
    if exclude_borders:
        print(
                '--> Number of particle(s) before border exclusion: ',
                str(n_particles))
        print('--> Excluding border particles...')
        labels = segmentation.clear_border(labels)
        # Calculate number of instances of each value in label_array 
        particleIDs = np.unique(labels)
        # Subtract 1 to account for background label
        n_particles = len(particleIDs) - 1
    print(
            f'--> Segmentation complete. '
            f'{n_particles} particle(s) segmented.')
    if print_size:
        # sys.getsizeof() doesn't represent nested objects; need to add manually
        print('--> Size of segmentation results (GB):')
        for key, val in segment_dict.items():
            print(f'----> {key}: {sys.getsizeof(val) / 1E9}')
    if return_dict:
        segment_dict = {
            'distance-map' : dist_map,
            'maxima' : maxima,
            'integer-labels' : labels,
        }
        return segment_dict
    else:
        return labels

def count_segmented_voxels(segment_dict, particleID=None, exclude_zero=True):
    """Count number of segmented voxels within particles of unique labels.

    Parameters
    ----------
    segment_dict : dict
        Dictionary containing segmentation routine steps, as returned from 
        watershed_segment(). Must contain key 'integer-labels'
    particleID : int or None, optional
        If an integer is passed, only the number of voxels within the particle 
        matching that integer are returned.
    exclude_zero : bool, optional
        Exclude zero label in count. Usually zero refers to background. 
        Defaults to True

    Returns
    -------
    If particleID is not None:
        int
            Number of voxels in particle labeled as particleID in 
            segmment_dict['integer-labels']
    Else:
        dict
            Dictionary with particleID keys and integer number of voxels 
            in particle corresponding to particleID key.
    """
    label_array = segment_dict['integer-labels']
    # Calculate number of instances of each value in label_array 
    particleIDs, nvoxels = np.unique(label_array, return_counts=True)
    nvoxels_by_ID_dict = dict(zip(particleIDs, nvoxels))
    if exclude_zero:
        del nvoxels_by_ID_dict[0]
    if particleID is not None:
        try:
            return nvoxels_by_ID_dict[particleID]
        except KeyError:
            raise ValueError(f'Particle ID {particleID} not found.')
    else:
        return nvoxels_by_ID_dict

def isolate_particle(segment_dict, particleID, erode=False):
    """Isolate a certain particle by removing all other particles in a 3D array.

    Parameters
    ----------
    segement_dict : dict
        Dictionary containing segmentation routine steps, as returned from 
        watershed_segment(), with at least the key: 'integer-labels' and 
        corresponding value: images with segmented particles labeled with 
        unique integers
    particleID : int
        Label corresponding to pixels in segment_dict['integer-labels'] that 
        will be plotted 
    erode : bool, optional
        If True, isolated particle will be eroded before array is returned.

    Returns
    -------
    numpy.ndarray
        3D array of the same size as segment_dict['integer-labels'] that is 
        only nonzero where pixels matched value of integer_label in original 
        array
    """
    imgs_single_particle = np.zeros_like(
        segment_dict['integer-labels'], dtype=np.uint8
    )
    imgs_single_particle[segment_dict['integer-labels'] == particleID] = 255
    if erode:
        imgs_single_particle = morphology.binary_erosion(imgs_single_particle)
    return imgs_single_particle

def save_stl(
    save_path, 
    verts, 
    faces, 
    spatial_res=1, 
    x_offset=0,
    y_offset=0,
    z_offset=0,
    suppress_save_message=False
):
    """Save triangular mesh defined by vertices and face indices as an STL file.

    Parameters
    ----------
    save_path : Path or str
        Path at which STL file will be saved. If doesn't end with '.stl', 
        it will be added.
    verts : array-like
        Array of (x, y, z) vertices indexed with faces to construct triangles.
    faces : array-like
        Array of indices referencing verts that define the triangular faces 
        of the mesh.
    spatial_res : float, optional
        Factor to apply to multiply spatial vectors of saved STL. Applying the 
        spatial/pixel resolution of the CT scan will give the STL file units of
        the value. Defaults to 1 to save the STL in units of pixels.
    x_offset : int, optional
        Integer value to offset x coordinates of STL. Related to column crop 
        and particel position.
    y_offset : int, optional
        Integer value to offset y coordinates of STL. Related to row crop and 
        particel position.
    z_offset : int, optional
        Integer value to offset z coordinates of STL. Related to slice crop and 
        particel position.
    suppress_save_message : bool, optional
        If True, particle label and STL file path will not be printed. By 
        default False
    """
    if not str(save_path).endswith('.stl'):
        save_path = Path(f'{save_path}.stl')
    if save_path.exists():
        print(f'File already exists: {save_path}')
    else:
        # Convert vertices (verts) and faces to numpy-stl format for saving:
        vertice_count = faces.shape[0]
        stl_mesh = mesh.Mesh(
            np.zeros(vertice_count, dtype=mesh.Mesh.dtype),
            remove_empty_areas=False
        )
        for i, face in enumerate(faces):
            # stl_mesh.vectors are the position vectors. Multiplying by the 
            # spatial resolution of the scan makes these vectors physical.
            # x coordinate (vector[0]) from col (face[2])
            stl_mesh.vectors[i][0] = spatial_res * verts[face[2], :]
            # y coordinate (vector[1]) from row (face[1])
            stl_mesh.vectors[i][1] = spatial_res * verts[face[1], :]
            # z coordinate (vector[2]) from slice (face[0])
            stl_mesh.vectors[i][2] = spatial_res * verts[face[0], :]
            translate_v = spatial_res * np.array([x_offset, y_offset, z_offset])
            stl_mesh.translate(translate_v)
        # Write the mesh to STL file
        stl_mesh.save(save_path)
        if not suppress_save_message:
            print(f'STL saved: {save_path}')

def create_surface_mesh(
        imgs, 
        slice_crop=None, 
        row_crop=None, 
        col_crop=None,
        min_slice=None, 
        min_row=None, 
        min_col=None, 
        spatial_res=1, 
        voxel_step_size=1,
        save_path=None,
):
    verts, faces, normals, values = measure.marching_cubes(
        imgs, step_size=voxel_step_size,
        allow_degenerate=False
    )
    # Flip vertices such that (slice, row, col)/(z, y, x) orientation 
    # becomes (x, y, z)
    verts = np.flip(verts, axis=1)
    # Convert vertices (verts) and faces to numpy-stl format for saving:
    vertice_count = faces.shape[0]
    stl_mesh = mesh.Mesh(
        np.zeros(vertice_count, dtype=mesh.Mesh.dtype),
        remove_empty_areas=False
    )
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[face[j], :]
    # Calculate offsets for STL coordinates
    if col_crop is not None:
        x_offset = col_crop[0]
    else: 
        x_offset = 0
    if row_crop is not None:
        y_offset = row_crop[0]
    else: 
        y_offset = 0
    if slice_crop is not None:
        z_offset = slice_crop[0]
    else:
        z_offset = 0
    # Add offset related to particle location. If min slice/row/col is 
    # provided, it's assumed to be provided from a voxel-padded array so  
    # the -1 accounts for the voxel padding on front end of each dimension.
    # If a min is not provided, the offset is calculated as the min nonzero 
    # voxel location in each dimension.
    if min_slice == None:
        z_offset += np.where(imgs)[0].min()
    else:
        z_offset += min_slice - 1
    if min_row == None:
        y_offset += np.where(imgs)[1].min()
    else:
        y_offset += min_row - 1
    if min_col == None:
        x_offset += np.where(imgs)[2].min()
    else:
        x_offset += min_col - 1
    # Apply offsets to (x, y, z) coordinates of mesh
    stl_mesh.x += x_offset
    stl_mesh.y += y_offset
    stl_mesh.z += z_offset
    # stl_mesh.vectors are the position vectors. Multiplying by the 
    # spatial resolution of the scan makes these vectors physical.
    stl_mesh.vectors *= spatial_res
    # Save STL if save_path provided
    if save_path is not None:
        stl_mesh.save(save_path)
    return verts, faces, normals, values

def save_as_stl_files(
    segmented_images,
    stl_dir_location,
    output_filename_base,
    suppress_save_msg=True,
    slice_crop=None,
    row_crop=None,
    col_crop=None,
    stl_overwrite=False,
    spatial_res=1,
    n_erosions=None,
    median_filter_voxels=True,
    voxel_step_size=1,
):
    """Iterate through particles in the regions list provided by 
    skimage.measure.regionprops()

    Parameters
    ----------
    segmented_images : numpy.ndarray
        3D DxMxN array representing D segmented images with M rows and N 
        columns. Each pixel/voxel of each particle is assigned a different 
        integer label to differentiate from neighboring and potentially 
        connected particles. Stored in "segment_dict['integer-labels']".
    stl_dir_location : Path or str
        Path to the directory where the STL files will be saved.
    suppress_save_msg : bool, optional
        If True, save messages are not printed for each STL. Defaults to True.
    slice_crop : list or None, optional
        Min and max crop in the slice dimension.
    row_crop : list or None, optional
        Min and max crop in the row dimension.
    col_crop : list or None, optional
        Min and max crop in the column dimension.
    spatial_res : float, optional
        Factor to apply to multiply spatial vectors of saved STL. Applying the 
        spatial/pixel resolution of the CT scan will give the STL file units of 
        the value. Defaults to 1 to save the STL in units of pixels.
    voxel_step_size : int, optional
        Number of voxels to iterate across in marching cubes algorithm. Larger 
        steps yield faster but coarser results. Defaults to 1. 
    allow_degenerate_tris : bool, optional
        Whether to allow degenerate (i.e. zero-area) triangles in the 
        end-result. If False, degenerate triangles are removed, at the cost of 
        making the algorithm slower. Defaults to False.
    n_erosions : int, optional
        Number of time morphologic erosion is applied to remove one layer of 
        voxels from outer layer of particle. Analagous to peeling the outer 
        skin of an onion. Defaults to False.
    print_index_extrema : bool, optional
        If True, list of the min/max of the slice, row, and column indices for 
        each saved particle are recorded and the ultimate min/max are printed 
        at the end of the function. Defaults to True.
    return_n_saved : bool, optional
        If True, the number of particles saved will be returned.

    Returns
    -------
    If return_dir_path is True:
        int
            Number of STL files saved.

    Raises
    ------
    ValueError
        Raise ValueError when directory named dir_name already exists at 
        location save_dir_parent_path
    """
    print('Generating surface meshes...')
    props_df = pd.DataFrame(columns=[
            'particleID',
            'meshed',
            'n_voxels',
            'centroid',
            'min_slice',
            'max_slice',
            'min_row',
            'max_row',
            'min_col',
            'max_col',])
    if n_erosions is None:
        n_erosions = 0
    regions = measure.regionprops(segmented_images)
    n_particles = len(regions)
    n_particles_digits = len(str(n_particles))
    for region in regions:
        # Create save path
        fn = (
            f'{output_filename_base}'
            f'{str(region.label).zfill(n_particles_digits)}.stl'
        )
        stl_save_path = Path(stl_dir_location) / fn
        # If STL can be saved, continue with process
        if stl_save_path.exists() and not stl_overwrite:
            raise ValueError(f'STL already exists: {stl_save_path}')
        elif not Path(stl_dir_location).exists():
            # Make directory if it doesn't exist
            Path(stl_dir_location).mkdir(parents=True)
        # Get bounding slice, row, and column
        min_slice, min_row, min_col, max_slice, max_row, max_col = region.bbox
        # Get centroid coords in slice, row, col and reverse to get x, y, z
        centroid_xyz = ', '.join(
            reversed([str(round(coord)) for coord in region.centroid])
        )
        props = {}
        props['particleID'] = region.label
        props['n_voxels']   = region.area
        props['centroid']   = centroid_xyz
        props['min_slice']  = min_slice
        props['max_slice']  = max_slice
        props['min_row']    = min_row
        props['max_row']    = max_row
        props['min_col']    = min_col
        props['max_col']    = max_col
        # If particle has less than 2 voxels in each dim, do not mesh surface
        # (marching cubes limitation)
        if (
            max_slice - min_slice <= 2 + 2*n_erosions
            and max_row - min_row <= 2 + 2*n_erosions
            and max_col - min_col <= 2 + 2*n_erosions
        ):
            props['meshed'] = False
            print(
                f'Surface mesh not created for particle {region.label}: '
                'Particle smaller than minimum width in at least one dimension.'
            )
        # Continue with process if particle has at least 2 voxels in each dim
        else:
            # Isolate Individual Particles
            imgs_particle = region.image
            imgs_particle_padded = np.pad(imgs_particle, 1)
            # Insert region inside padding
            imgs_particle_padded[1:-1, 1:-1, 1:-1] = imgs_particle
            if n_erosions is not None and n_erosions > 0:
                for _ in range(n_erosions):
                    imgs_particle_padded = morphology.binary_erosion(
                        imgs_particle_padded
                    )
                particle_labeled = measure.label(
                    imgs_particle_padded, connectivity=1
                )
                particle_regions = measure.regionprops(particle_labeled)
                if len(particle_regions) > 1:
                    # Sort particle regions by area with largest first
                    particle_regions = sorted(
                        particle_regions, key=lambda r: r.area, reverse=True
                    )
                    # Clear non-zero voxels from imgs_particle_padded
                    imgs_particle_padded = np.zeros_like(
                        imgs_particle_padded, dtype=np.uint8
                    )
                    # Add non-zero voxels back for voxels belonging to largest 
                    # particle present (particle_regions[0])
                    imgs_particle_padded[
                        particle_labeled == particle_regions[0].label
                    ] = 255  # (255 is max for 8-bit/np.uint8 image)
            if median_filter_voxels:
                # Median filter used to smooth particle in image/voxel form
                imgs_particle_padded = filters.median(imgs_particle_padded)
            # Perform marching cubes surface meshing when array has values > 0
            try:
                vertices, faces, normals, vals = create_surface_mesh(
                        imgs_particle_padded, slice_crop=slice_crop, 
                        row_crop=row_crop, col_crop=col_crop, 
                        min_slice=min_slice, min_row=min_row, min_col=min_col, 
                        spatial_res=spatial_res, 
                        voxel_step_size=voxel_step_size, 
                        save_path=stl_save_path)
                props['meshed'] = True
                if not suppress_save_msg:
                    print(f'STL saved: {stl_save_path}')
            except RuntimeError as error:
                props['meshed'] = False
                print(
                    f'Surface mesh not created for particle {region.label}. '
                    'Particle likely too small. Error: ',
                    error
                )
        props_df = pd.concat(
            [props_df, pd.DataFrame.from_records([props])], ignore_index=True
        )
    csv_fn = (f'{output_filename_base}properties.csv')
    csv_save_path = Path(stl_dir_location) / csv_fn
    props_df.to_csv(csv_save_path, index=False)
    # Count number of meshed particles
    n_saved = len(np.argwhere(props_df['meshed'].to_numpy()))
    print(f'--> {n_saved} STL file(s) written!')

def check_properties(mesh):
    n_triangles = len(mesh.triangles)
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()
    print(f"  n_triangles:            {n_triangles}")
    print(f"  watertight:             {watertight}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  orientable:             {orientable}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print()

def repair_mesh(stl_mesh):
    stl_mesh.remove_degenerate_triangles()
    stl_mesh.remove_duplicated_triangles()
    stl_mesh.remove_duplicated_vertices()
    stl_mesh.remove_non_manifold_edges()
    return stl_mesh

def simplify_mesh(
    stl_mesh, n_tris, recursive=False, failed_iter=10
):
    simplified_mesh = stl_mesh.simplify_quadric_decimation(n_tris)
    stl_mesh = repair_mesh(stl_mesh)
    stl_mesh.compute_triangle_normals()
    stl_mesh.compute_vertex_normals()
    if recursive and not simplified_mesh.is_watertight():
        simplified_mesh, n_tris = simplify_mesh(
            stl_mesh, n_tris + failed_iter, recursive=True
        )
    return simplified_mesh, n_tris
    
def simplify_mesh_iterative(
    stl_mesh, target_n_tris, return_mesh=True, iter_factor=2, 
    suppress_save_msg=True
):
    og_n_tris = len(stl_mesh.triangles)
    prev_n_tris = len(stl_mesh.triangles)
    n_iters = 0
    while prev_n_tris > target_n_tris:
        stl_mesh, n_tris = simplify_mesh(stl_mesh, prev_n_tris // iter_factor)
        if n_tris == prev_n_tris:
            break
        prev_n_tris = n_tris
        n_iters += 1
    if not suppress_save_msg:
        print(
            f'Mesh simplified: {og_n_tris} -> {len(stl_mesh.triangles)}'
            f' in {n_iters} iterations'
        )
    if return_mesh:
        return stl_mesh 

def postprocess_mesh(
        stl_save_path, 
        smooth_iter=1, 
        simplify_n_tris=250, 
        iterative_simplify_factor=None, 
        recursive_simplify=False,
        resave_mesh=False):
    stl_save_path = str(stl_save_path)
    stl_mesh = o3d.io.read_triangle_mesh(stl_save_path)
    stl_mesh = repair_mesh(stl_mesh)
    if smooth_iter is not None:
        stl_mesh = stl_mesh.filter_smooth_laplacian(
            number_of_iterations=smooth_iter
        )
    if simplify_n_tris is not None:
        if iterative_simplify_factor is not None:
            stl_mesh = simplify_mesh_iterative(
                stl_mesh, simplify_n_tris, iter_factor=iterative_simplify_factor
            )
        else:
            stl_mesh, n_tris = simplify_mesh(
                stl_mesh, simplify_n_tris, recursive=recursive_simplify, 
                failed_iter=1
            )
    if resave_mesh:
        stl_mesh.compute_triangle_normals()
        stl_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            stl_save_path, stl_mesh, 
            # Currently unsupported to save STLs in ASCII format
            # write_ascii=True
        )
    mesh_props = {}
    mesh_props['n_triangles'] = len(stl_mesh.triangles)
    mesh_props['watertight'] = stl_mesh.is_watertight()
    mesh_props['self_intersecting'] = stl_mesh.is_self_intersecting()
    mesh_props['orientable'] = stl_mesh.is_orientable()
    mesh_props['edge_manifold'] = stl_mesh.is_edge_manifold(allow_boundary_edges=True)
    mesh_props['edge_manifold_boundary'] = stl_mesh.is_edge_manifold(allow_boundary_edges=False)
    mesh_props['vertex_manifold'] = stl_mesh.is_vertex_manifold()
    return stl_mesh, mesh_props

def postprocess_meshes(
        stl_save_path, 
        smooth_iter=None, 
        simplify_n_tris=None, 
        iterative_simplify_factor=None, 
        recursive_simplify=False,
        resave_mesh=False):
    print('Postprocessing surface meshes...')
    # Iterate through each STL file, load the mesh, and smooth/simplify
    for i, stl_path in enumerate(Path(stl_save_path).glob('*.stl')):
        stl_mesh, mesh_props = postprocess_mesh(
                stl_path, 
                smooth_iter=smooth_iter,
                simplify_n_tris=simplify_n_tris,
                iterative_simplify_factor=iterative_simplify_factor,
                recursive_simplify=recursive_simplify, 
                resave_mesh=resave_mesh)
        # props = {**props, **mesh_props}
    try:
        print(f'--> {i + 1} surface meshes postprocessed.')
    except NameError:
        print('No meshes found to postprocess.')

def save_images(
    imgs,
    save_dir,
    img_names=None,
    convert_to_16bit=False
):
    """Save images to save_dir.

    Parameters
    ----------
    imgs : numpy.ndarray or list
        Images to save, either as a list or a 3D numpy array (4D array of 
        colored images also works)
    save_dir : str or Path
        Path to new directory to which iamges will be saved. Directory must 
        not already exist to avoid accidental overwriting. 
    img_names : list, optional
        List of strings to be used as image filenames when saved. If not 
        included, images will be names by index. Defaults to None.
    convert_to_16bit : bool, optional
        Save images as 16-bit, by default False
    """
    save_dir = Path(save_dir)
    # Create directory, or raise an error if that directory already exists
    save_dir.mkdir(parents=True, exist_ok=False)
    # If imgs is a numpy array and not a list, convert it to a list of images
    if isinstance(imgs, np.ndarray):
        # If 3D: (slice, row, col)
        if len(imgs.shape) == 3:
            file_suffix = 'tif'
            imgs = [imgs[i, :, :] for i in range(imgs.shape[0])]
        # If 4D: (slice, row, col, channel) where channel is RGB (color) value
        elif len(imgs.shape) == 4:
            file_suffix = 'png'
            imgs = [
                util.img_as_ubyte(imgs[i, :, :, :]) 
                for i in range(imgs.shape[0])
            ]
    for i, img in enumerate(imgs):
        if convert_to_16bit:
            img = img.astype(np.uint16)
        # if no img_names, use the index of the image
        if img_names is None:
            img_name = str(i).zfill(3)
        else:
            img_name = img_names[i]
        iio.imsave(Path(save_dir / f'{img_name}.{file_suffix}'), img)
    print(f'{len(imgs)} image(s) saved to: {save_dir.resolve()}')

#~~~~~~~~~~~~~~~~~~~
# Plotting Functions
#~~~~~~~~~~~~~~~~~~~

def plot_mesh_3D(verts, faces):
    """Plot triangualar mesh with Matplotlib.

    Parameters
    ----------
    verts : array-like
        Array of (x, y, z) vertices indexed with faces to construct triangles.
    faces : array-like
        Array of indices referencing verts that define the triangular faces of 
        the mesh.

    Returns
    -------
    matplotlib.figure, matplotlib.axis
        Matplotlib figure and axis objects corresponding to 3D plot
    """
    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('black')
    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_xlim(min(verts[:, 0]), max(verts[:, 0]))
    ax.set_ylim(min(verts[:, 1]), max(verts[:, 1]))
    ax.set_zlim(min(verts[:, 2]), max(verts[:, 2]))
    return fig, ax

def plot_stl(path_or_mesh, zoom=True):
    """Load an STL and plot it using matplotlib.

    Parameters
    ----------
    stl_path : Path or str
        Path to an STL file to load or a directory containing STL. If directory, 
        a random STL file will be loaded from the directory.
    zoom : bool, optional
        If True, plot will be zoomed in to show the particle as large as  
        Defaults to True

    Returns
    -------
    matplotlib.figure, matplotlib.axis
        Matplotlib figure and axis objects corresponding to 3D plot
    """
    if isinstance(path_or_mesh, str) or isinstance(path_or_mesh, Path):
        stl_path = Path(path_or_mesh)
        # If stl_path is a directory, choose a random file from inside
        if stl_path.is_dir():
            stl_path_list = [path for path in Path(stl_path).glob('*.stl')]
            if len(stl_path_list) == 0:
                raise ValueError(f'No STL files found in directory: {stl_path}')
            random_i = np.random.randint(0, len(stl_path_list))
            stl_path = stl_path_list[random_i]
            print(f'Plotting STL: {stl_path.name}')
        elif not str(stl_path).endswith('.stl'):
            raise ValueError(f'File is not an STL: {stl_path}')
        # Load the STL files and add the vectors to the plot
        stl_mesh = mesh.Mesh.from_file(stl_path)
    elif isinstance(path_or_mesh, mesh.Mesh):
        stl_mesh = path_or_mesh
    else:
        raise ValueError(
            f'First parameter must string, pathlib.Path, or stl.mesh.Mesh object. '
            f'Object type: {type(path_or_mesh)}'
        )
    mpl_mesh = Poly3DCollection(stl_mesh.vectors)
    mpl_mesh.set_edgecolor('black')
    # Display resulting triangular mesh using Matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(mpl_mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    if zoom:
        # stl_mesh.vectors is Mx3x3 array (M 2D (3x3) arrays: [x, y, z]), 
        # Transpose (array.T) is 3x3xM array (3 2D (3xM) arrays: [x], [y], [z])
        ax.set_xlim(np.min(stl_mesh.vectors.T[0]), \
            np.max(stl_mesh.vectors.T[0]))
        ax.set_ylim(np.min(stl_mesh.vectors.T[1]), \
            np.max(stl_mesh.vectors.T[1]))
        ax.set_zlim(np.min(stl_mesh.vectors.T[2]), \
            np.max(stl_mesh.vectors.T[2]))
    return fig, ax

def plot_particle_slices(imgs_single_particle, n_slices=4, fig_w=7, dpi=100):
    """Plot a series of images of a single particle across n_slices number of 
    slices.

    Parameters
    ----------
    imgs_single_particle : numpy.ndarray
        3D array of the same size as segment_dict['integer-labels'] that is 
        only nonzero where pixels matched value of integer_label in original 
        array
    n_slices : int, optional
        Number of slices to plot as images in the figure, by default 4
    fig_w : int, optional
        Width of figure in inches, by default 7

    Returns
    -------
    matplotlib.figure, matplotlib.axis
        Matplotlib figure and axis objects corresponding to 3D plot
    """
    # bounds: (min_slice, min_row, min_col, max_slice, max_row, max_col)
    bounds = measure.regionprops(imgs_single_particle)[0].bbox
    print(f'Particle bounds: {bounds[0], bounds[3]}, {bounds[1], bounds[4]}, \
        {bounds[2], bounds[5]}')

    # bounds[0] and bounds[3] used for min_slice and max_slice respectively
    slices = [round(i) for i in np.linspace(bounds[0], bounds[3], n_slices)]
    n_axes_h = 1
    n_axes_w = n_slices
    img_w = imgs_single_particle.shape[2]
    img_h = imgs_single_particle.shape[1]
    title_buffer = .5
    fig_h = fig_w * (img_h / img_w) * (n_axes_h / n_axes_w) + title_buffer
    fig, axes = plt.subplots(
        n_axes_h, n_axes_w, dpi=dpi, figsize=(fig_w, fig_h), 
        constrained_layout=True, facecolor='white',
    )
    if not isinstance(axes, np.ndarray):
        ax = [axes]
    else:
        ax = axes.ravel()
    for i, slice_i in enumerate(slices):
        ax[i].imshow(imgs_single_particle[slice_i, ...],interpolation='nearest')
        ax[i].set_axis_off()
        ax[i].set_title(f'Slice: {slice_i}')
    return fig, ax

def plot_imgs(
    imgs, 
    n_imgs=3, 
    slices=None, 
    print_slices=True,
    imgs_per_row=None, 
    fig_w=7.5, 
    dpi=100
):
    """Plot images.

    Parameters
    ----------
    imgs : list
        3D NumPy array or list of 2D arrays representing images to be plotted.
    fig_w : float, optional
        Width of figure in inches, by default 7.5 
    n_imgs : int, optional
        Number of slices to plot from 3D array. Defaults to 3.
    slices : None or list, optional
        Slice numbers to plot. Replaces n_imgs. Defaults to None.
    print_slices : bool, optional
        If True, print the slices being plotted. Defaults to True.
    dpi : float, optional
        Resolution (dots per inch) of figure. Defaults to 300.

    Returns
    -------
    matplotlib.Figure, matplotlib.Axis
        2-tuple containing matplotlib figure and axes objects
    """
    dim = len(imgs.shape)
    if dim == 2:
        n_imgs = 1
        total_imgs = 1
        img_w = imgs.shape[1]
        img_h = imgs.shape[0]
    else:
        total_imgs = imgs.shape[0]
        img_w = imgs[0].shape[1]
        img_h = imgs[0].shape[0]
    if slices is None:
        spacing = total_imgs // n_imgs
        img_idcs = [i * spacing for i in range(n_imgs)]
    else:
        n_imgs = len(slices)
        img_idcs = slices
    if imgs_per_row is None:
        n_cols = n_imgs
    else: 
        n_cols = imgs_per_row
    n_rows = int(math.ceil( n_imgs / n_cols ))
    fig_h = fig_w * (img_h / img_w) * (n_rows / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True, dpi=dpi, 
        facecolor='white'
    )
    if n_imgs == 1:
        axes.imshow(imgs, interpolation='nearest')
        axes.axis('off')
    else:
        ax = axes.ravel()
        if print_slices:
            print(f'Plotting images: {img_idcs}')
        for i, idx in enumerate(img_idcs):
            ax[i].imshow(imgs[idx, ...], interpolation='nearest')
        # Separated from loop in the that axes are left blank (un-full row)
        for a in ax:
            a.axis('off')
    return fig, axes

def plot_particle_labels(
    segment_dict, 
    img_idx,
    label_color='white',
    label_bg_color=(0, 0, 0, 0),
    use_color_labels=True,
    fig_w=7,
    dpi=100,
):
    """Plot segmented particles 

    Parameters
    ----------
    segment_dict : dict
        Dictionary containing segmentation routine steps, as returned 
        from watershed_segment()
    img_idx : int
        Index of image on which particle labels will be shown
    label_color : str, optional
        Color of text of which labels will be shown, by default 'white'
    label_bg_color : tuple, optional
        Color of label background. Defaults to transparent RGBA tuple: 
        (0, 0, 0, 0)
    use_color_labels : bool, optional
        If true, labels are converted to color labels to be plotted on image.
    fig_w : int, optional
        Width in inches of figure that will contain the labeled image, by 
        default 7

    Returns
    -------
    matplotlib.figure, matplotlib.axis
        Matplotlib figure and axis objects corresponding to 3D plot
    """
    labels = segment_dict['integer-labels']
    regions = measure.regionprops(labels[img_idx, ...])
    label_centroid_pairs = [(region.label, region.centroid) \
    for region in regions]
    n_axes_h = 1
    n_axes_w = 1
    img_w = labels.shape[2]
    img_h = labels.shape[1]
    title_buffer = .5
    fig_h = fig_w * (img_h / img_w) * (n_axes_h / n_axes_w) + title_buffer
    fig, ax = plt.subplots(
        n_axes_h, n_axes_w, dpi=dpi, figsize=(fig_w, fig_h), 
        constrained_layout=True, facecolor='white',
    )
    if use_color_labels:
        labels = color.label2rgb(labels)
    ax.imshow(labels[img_idx, ...], interpolation='nearest')
    ax.set_axis_off()
    for label, centroid in label_centroid_pairs:
        ax.text(
            centroid[1], centroid[0], str(label), fontsize='large',
            color=label_color, backgroundcolor=label_bg_color, ha='center', 
            va='center'
        )
    return fig, ax

def plot_segment_steps(
    imgs, 
    imgs_pre, 
    imgs_binarized, 
    segment_dict, 
    n_imgs=3, 
    slices=None,
    plot_maxima=True,
    fig_w=7.5, 
    dpi=100
):
    """Plot images.

    Parameters
    ----------
    imgs : list
        3D NumPy array or list of 2D arrays representing images to be plotted.
    imgs_pre : list
        3D NumPy array or list of 2D arrays representing preprocessed images to
        be plotted.
    imgs_binarized : list
        3D NumPy array or list of 2D arrays representing binarized images to be 
        plotted.
    segment_dict : dict
        Dictionary containing segmentation routine steps, as returned from 
        watershed_segment().
    n_imgs : int, optional
        Number of 2D images to plot from the 3D array. Defaults to 3.
    plot_maxima : int, optional
        If true, maxima used to seed watershed segmentation will be plotted on 
        distance map. Defaults to True.
    fig_w : float, optional
        Width of figure in inches, by default 7.5 
    dpi : float, optional
        Resolution (dots per inch) of figure. Defaults to 300.

    Returns
    -------
    matplotlib.Figure, matplotlib.Axis
        2-tuple containing matplotlib figure and axes objects
    """
    dim = len(imgs.shape)
    if dim == 2:
        n_imgs = 1
        total_imgs = 1
        img_w = imgs.shape[1]
        img_h = imgs.shape[0]
    else:
        total_imgs = imgs.shape[0]
        img_w = imgs[0].shape[1]
        img_h = imgs[0].shape[0]
    if slices is None:
        spacing = total_imgs // n_imgs
        img_idcs = [i * spacing for i in range(n_imgs)]
    else:
        n_imgs = len(slices)
        img_idcs = slices
    n_axes_h = n_imgs
    n_axes_w = 5
    fig_h = fig_w * (img_h / img_w) * (n_axes_h / n_axes_w)
    fig, axes = plt.subplots(
        n_axes_h, n_axes_w, figsize=(fig_w, fig_h), constrained_layout=True, 
        dpi=dpi, facecolor='white'
    )
    if n_imgs == 1:
        axes.imshow(imgs, interpolation='nearest')
        axes.axis('off')
    else:
        spacing = total_imgs // n_imgs
        print(f'Plotting images: {img_idcs}')
        for i in range(n_imgs):
            idx = img_idcs[i]
            # Plot the raw image
            axes[i, 0].imshow(imgs[idx, ...], interpolation='nearest')
            # Plot the preprocessed image
            axes[i, 1].imshow(imgs_pre[idx, ...], interpolation='nearest')
            # Plot the binarized image
            axes[i, 2].imshow(imgs_binarized[idx, ...], interpolation='nearest')
            # Plot the distance map image
            axes[i, 3].imshow(
                segment_dict['distance-map'][idx, ...], interpolation='nearest'
            )
            # Plot the maxima
            if plot_maxima:
                x = segment_dict['maxima'][:, 2]
                y = segment_dict['maxima'][:, 1]
                # Find the maxima that fall on the current slice (img_idx)
                x_img_idx = x[segment_dict['maxima'][:, 0] == idx]
                y_img_idx = y[segment_dict['maxima'][:, 0] == idx]
                axes[i, 3].scatter(x_img_idx, y_img_idx, color='red', s=1)
            # Convert the integer labels to colored labels and plot
            int_labels = segment_dict['integer-labels'][idx, ...]
            color_labels = color.label2rgb(int_labels, bg_label=0)
            axes[i, 4].imshow(color_labels, interpolation='nearest')
    for a in axes.ravel():
        a.set_axis_off()
    return fig, axes
    
#~~~~~~~~~
# Workflow
#~~~~~~~~~

def segmentation_workflow(argv):
    #---------------------------
    # Get command-line arguments
    #---------------------------
    try:
        opts, args = getopt.getopt(argv,"hf:",["ifile=","ofile="])
    except getopt.GetoptError:
        fatalError('Error in command-line arguments.  \
            Enter ./segment.py -h for more help')
    yaml_file = ''
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        if opt == "-f":
            yaml_file = str(arg)

    #---------------------
    # Read YAML input file
    #---------------------
    if yaml_file == '':
        fatalError(
            'No input file specified. Try ./segment.py -h for more help.'
    )
    else:
        # Load YAML inputs into a dictionary
        ui = load_inputs(yaml_file)

    #------------
    # Load images
    #------------
    print()
    imgs = load_images(
        ui['ct_img_dir'],
        slice_crop=ui['slice_crop'],
        row_crop=ui['row_crop'],
        col_crop=ui['col_crop'],
        convert_to_float=True,
        file_suffix=ui['file_suffix']
    )

    #------------------
    # Preprocess images
    #------------------
    print()
    imgs_pre = preprocess(
        imgs, median_filter=ui['pre_seg_med_filter'], 
        rescale_intensity_range=ui['rescale_range']
    )

    #----------------
    # Binarize images
    #----------------
    print()
    imgs_binarized, thresh_vals = binarize_multiotsu(
        imgs_pre, n_otsu_classes=ui['n_otsu_classes'], 
        n_selected_thresholds=ui['n_selected_classes'], 
    )

    #---------------
    # Segment images
    #---------------
    print()
    segment_dict = watershed_segment(
        imgs_binarized, min_peak_distance=ui['min_peak_dist'], 
        use_int_dist_map=ui['use_int_dist_map'], 
        exclude_borders=ui['exclude_borders'], return_dict=True
    )
    
    if ui['create_stls']:
        #---------------------------------------
        # Create Surface Meshes of Each Particle 
        #---------------------------------------
        print()
        save_as_stl_files(
            segment_dict['integer-labels'],
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

        #---------------------------------------------
        # Postprocess surface meshes for each particle
        #---------------------------------------------
        if (
                ui['mesh_smooth_n_iters'] is not None
                or ui['mesh_simplify_n_tris'] is not None
                or ui['mesh_simplify_factor'] is not None):
            print()
            # Iterate through each STL file, load the mesh, and smooth/simplify
            postprocess_meshes(
                    ui['stl_dir_location'], 
                    smooth_iter=ui['mesh_smooth_n_iters'], 
                    simplify_n_tris=ui['mesh_simplify_n_tris'], 
                    iterative_simplify_factor=ui['mesh_simplify_factor'], 
                    recursive_simplify=False, resave_mesh=True)

    #------------------------
    # Plot figures if enabled
    #------------------------
    if ui['seg_fig_show']:
        fig_seg_steps, axes_seg_steps = plot_segment_steps(
                imgs, imgs_pre, imgs_binarized, segment_dict, 
                n_imgs=ui['seg_fig_n_imgs'], slices=ui['seg_fig_slices'], 
                plot_maxima=ui['seg_fig_plot_max'])
    if ui['label_fig_show']:
        fig_labels, ax_labels = plot_particle_labels(
                segment_dict, ui['label_fig_idx'])
    if ui['stl_fig_show']:
        fig_stl, ax_stl = plot_stl(ui['stl_dir_location'])
    if ui['seg_fig_show'] or ui['label_fig_show'] or ui['stl_fig_show']:
        plt.show()
    

if __name__ == '__main__':
    print('')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('')
    print('Beginning Segmentation Workflow')
    print('')
    segmentation_workflow(sys.argv[1:])
    print('')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Successful Completion. Bye!')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('')
    print()
        
