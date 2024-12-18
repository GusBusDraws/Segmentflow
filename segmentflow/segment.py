#~~~~~~~~~~#
# Packages #
#~~~~~~~~~~#
import imageio.v3 as iio
import math
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from pathlib import Path
import pandas as pd
import scipy
from scipy import ndimage, spatial
from skimage import (
        draw, exposure, feature, filters, morphology, measure,
        segmentation, transform, util)
import stl
import sys
import trimesh
import yaml


#~~~~~~~~~~~#
# Functions #
#~~~~~~~~~~~#
def binarize_3d(
    imgs,
    thresh_val=0.65,
    fill_holes=64,
    return_process_dict=False
):
    """Creates binary images from list of images using a threshold value.
    ----------
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
    -------
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
            filled[i, :, :] = ndimage.binary_fill_holes(binarized[i, :, :])
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

def binarize_multiotsu(
    imgs,
    n_otsu_classes=2,
    downsample_image_factor=1,
    n_selected_thresholds=1,
    exclude_borders=False,
    print_size=False,
):
    """Binarize stack of images (3D array) using multi-Otsu thresholding
    algorithm.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing slices of a 3D volume.
    n_otsu_classes : int, optional
        Number of classes to threshold images, by default 2
    imgs_downsample_factor : int, optional
        Factor by which 3D images will be downsized (N) to speed up
        multi-Otsu calculation by only using every Nth 2D image from the
        stack in the calulcation.
        Defaults to 1 to use every image (i.e. no downsampling)
    n_selected_thresholds : int, optional
        Number of classes to group together (from the back of the thresholded
        values array returned by multi-Otsu function) to create binary image,
        by default 1
    exclude_borders : bool, optional
        If True, exclude particles that touch the border of the volume chunk
        specified by slice/row/col crop in load_images(). Defaults to False.
    print_size : bool, optional
        If True, print size of binarized images in GB. Defaults to False.
    -------
    Returns
    -------
    numpy.ndarray, list
        3D array of the shape imgs.shape containing binarized images; list of
        threshold values used to create binarized images
    """
    print('Binarizing images...')
    imgs_binarized = np.zeros_like(imgs, dtype=np.uint8)
    print('--> Calculating Otsu threshold(s)...')
    if downsample_image_factor > 1:
        imgs = imgs[::downsample_image_factor]
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
        silence=False,
        logger=None
):
    if not silence:
        log(logger, 'Creating surface mesh with marching cubes algorithm...')
    verts, faces, normals, values = measure.marching_cubes(
        imgs, step_size=voxel_step_size,
        allow_degenerate=False
    )
    if not silence:
        log(logger, 'Converting mesh to STL format...')
    # Flip vertices such that (slice, row, col)/(z, y, x) orientation
    # becomes (x, y, z)
    verts = np.flip(verts, axis=1)
    # Convert vertices (verts) and faces to numpy-stl format for saving:
    vertice_count = faces.shape[0]
    stl_mesh = stl.mesh.Mesh(
        np.zeros(vertice_count, dtype=stl.mesh.Mesh.dtype),
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
    stl_mesh_x, stl_mesh_y, stl_mesh_z = stl_mesh.x, stl_mesh.y, stl_mesh.z
    if save_path is not None:
        stl_mesh.save(save_path)
    return stl_mesh_x, stl_mesh_y, stl_mesh_z

def calc_voxel_stats(
    imgs_labeled,
    logger=None
):
    """Calculate the ratio of particle voxels (labels > 1)
    to binder voxels (labels = 0).
    ----------
    Parameters
    ----------
    imgs_labeled : numpy.ndarray
        DxMxN array where particles are labeled with integers greater than 1
        and binder is labeled as 1.
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    -------
    Returns
    -------
    float
        Floating point number representing ratio of the number of particle
        voxels to the number of binder voxels.
    """
    log(logger, 'Calculating voxel statistics...')
    n_voxels = imgs_labeled.shape[0] * imgs_labeled.shape[1] * imgs_labeled.shape[2]
    n_void = np.count_nonzero(imgs_labeled == 0)
    log(logger, f'--> Number of void voxels: {n_void}')
    n_binder = np.count_nonzero(imgs_labeled == 1)
    log(logger, f'--> Number of binder voxels: {n_binder}')
    n_particles = np.count_nonzero(imgs_labeled > 1)
    log(logger, f'--> Number of particle voxels: {n_particles}')
    n_remainder = n_voxels - n_void - n_binder - n_particles
    if n_remainder != 0:
        msg = (
            'WARNING: remainder detected between n_voxles, n_void, n_binder,'
            ' and n_particles'
        )
        log(logger, msg)
    particles_to_binder = n_particles / n_binder
    log(logger, f'--> Particle to binder volume ratio: {particles_to_binder}')
    return particles_to_binder

def fill_holes(imgs_semantic, logger=None):
    """Fill holes and smooth voxels in a semantic segmentation.
    ----------
    Parameters
    ----------
    imgs_semantic : numpy.ndarray
        DxMxN array where particles are labeled with 2
        and binder is labeled as 1.
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    -------
    Returns
    -------
    """
    log(logger, 'Filling holes...')
    imgs_particles = np.zeros_like(imgs_semantic, dtype=bool)
    # Create binary image matching location of particles
    imgs_particles[imgs_semantic == 2] = 1
    imgs_particles = ndimage.binary_fill_holes(imgs_particles)
    # Replace voxels in semantic matching new location of filled particles
    imgs_semantic[imgs_particles == 1] = 2
    # # Replace small features with surrounding space
    imgs_semantic = filters.median(imgs_semantic)
    return imgs_semantic

def fit_circle_to_edges(edge_img):
    # Perform a Hough Transform
    hough_radii = np.arange(edge_img.shape[0] // 4, edge_img.shape[0]//2, 2)
    hspaces = transform.hough_circle(edge_img, hough_radii)
    results = transform.hough_circle_peaks(hspaces, hough_radii, num_peaks=1)
    accum, center_x, center_y, radius = [row[0] for row in results]
    return center_x, center_y, radius

def generate_input_file(
        out_dir_path,
        workflow_name,
        categorized_input_shorthands,
        default_values
    ):
    print('Generating input file...')
    if out_dir_path.endswith('.yml') or out_dir_path.endswith('.yaml'):
        yaml_path = Path(out_dir_path)
    elif Path(out_dir_path).is_dir():
        yaml_path = Path(out_dir_path) / f'{workflow_name}_input.yml'
    shorthands = []
    params = []
    categorized_params = {}
    for category, pair in categorized_input_shorthands.items():
        categorized_params[category] = {}
        for shorthand, param in pair.items():
            categorized_params[category][param] = default_values[shorthand]
            shorthands.append(shorthand)
            params.append(param)
    with open(str(yaml_path), 'w') as file:
        doc = yaml.dump(categorized_params, file, sort_keys=False)
    print('--> Input file generated:', yaml_path.resolve())
    print()
    print('Exiting.')

def get_dims_df(imgs_labeled, logger=None):
    """Get dimension DataFrame for analyzing the size of particles based
    on the Cartesian bounding box of each particle.
    ----------
    Parameters
    ----------
    imgs_labeled : numpy.ndarray
        3D DxMxN array representing segmented images with pixels labeled
        corresponding to unique particle integers
    -------
    Returns
    -------
    pandas.DataFrame
        DataFrame object with the columns "nslices", "nrows", and "ncols"
    """
    msg = 'Collecting bounding box info...'
    print(msg) if logger is None else logger.info(msg)
    # Format segmented data
    dims_df = pd.DataFrame(measure.regionprops_table(
        imgs_labeled, properties=['label', 'area', 'bbox']))
    dims_df = dims_df.rename(columns={'area' : 'volume'})
    # Calculate nslices by subtracting z min from max
    dims_df['nslices'] = (
        dims_df['bbox-3'].to_numpy() - dims_df['bbox-0'].to_numpy())
    # Calculate nrows by subtracting y min from max
    dims_df['nrows'] = (
        dims_df['bbox-4'].to_numpy() - dims_df['bbox-1'].to_numpy())
    # Calculate ncols by subtracting x min from max
    dims_df['ncols'] = (
        dims_df['bbox-5'].to_numpy() - dims_df['bbox-2'].to_numpy())
    return dims_df

def help(workflow_name, workflow_desc):
    print()
    print('----------------------------------------------------------------')
    print()
    print(f'This is {workflow_name}.py, a workflow script for Segmentflow.')
    print()
    print(workflow_desc)
    print()
    print('----------------------------------------------------------------')
    print()
    print('Usage:')
    print()
    print(
        f'python -m segmentflow.workflows.{workflow_name}'
        ' -i path/to/input_file.yml'
    )
    print()
    print(
        'where input_file.yml is the YAML input file.'
        ' See the example input file at the top-level directory of the repo'
        ' to learn more about the content (inputs) of the input file.'
    )
    print()

def isolate_classes(
    imgs,
    threshold_values,
    intensity_step=1,
):
    """Threshold array with multiple threshold values to separate classes
    (semantic segmentation).
    ----------
    Parameters
    ----------
    imgs : list
        DxMxN array (D slices, M rows, N columns) NumPy array to be segmented
        according to threshold values.
    threshold_values : list or float
        Float or list of floats to segment image.
    intensity_step : int, optional
        Step value separating intensities. Defaults to 1, but might be set to
        soemthing like 125 such that isolated classes could be viewable in
        saved images.
    -------
    Returns
    -------
    numpy.ndarray
        DxMxN array representing semantic segmentation.
    """
    if not isinstance(threshold_values, list):
        threshold_values = [threshold_values]
    # Sort thresh_vals in ascending order then reverse to get largest first
    threshold_values.sort()
    imgs_semantic = np.zeros_like(imgs, dtype=np.uint8)
    # Starting with the lowest threshold value, set pixels above each
    # increasing threshold value to an increasing unique marker (1, 2, etc.)
    # multiplied by the intesnity_step parameter
    for i, val in enumerate(threshold_values):
        imgs_semantic[imgs > val] = int((i + 1) * intensity_step)
    return imgs_semantic

def isolate_particle(segment_dict, particleID, erode=False):
    """Isolate a certain particle by removing all other particles in a 3D array.
    ----------
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
    -------
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
    logger=None,
):
    """Load images from path and return as list of 2D arrays.
        Can also return names of images.
    ----------
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
        Defaults to 'tiff'
    print_size : bool, optional
        If True, print size of loaded images in GB. Defaults to False.
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    -------
    Returns
    -------
    list, numpy.ndarray, or tuple
        List of arrays or 3D array representing images
        (depending on return_3d_array), or if also_return_names is True,
        list containing names of images from filenames is also returned.
    ------
    Raises
    ------
    ValueError
        Raised when img_dir does not exist or is not a directory.
    """
    if logger is None:
        print('Loading images...')
    else:
        logger.info('Loading images...')
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise ValueError(f'Image directory not found: {img_dir}')
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
        imgs[i, ...] = iio.imread(img_path)[
            row_crop[0] : row_crop[1], col_crop[0] : col_crop[1]
        ]
    if convert_to_float:
        imgs = util.img_as_float(imgs)
    if logger is None:
        print('--> Images loaded as 3D array: ', imgs.shape)
    else:
        logger.info(f'--> Images loaded as 3D array: {imgs.shape}')
    if print_size:
        if logger is None:
            print('--> Size of array (GB): ', imgs.nbytes / 1E9)
        else:
            logger.info(f'--> Size of array (GB): {imgs.nbytes / 1E9}')
    if also_return_names and also_return_dir_name:
        return imgs, [img_path.stem for img_path in img_path_list], img_dir.stem
    elif also_return_names:
        return imgs, [img_path.stem for img_path in img_path_list]
    elif also_return_dir_name:
        return imgs, img_dir.stem
    else:
        return imgs

def load_inputs(
    yaml_path,
    categorized_input_shorthands,
    default_values,
):
    """Load input file and output a dictionary filled with default values
    for any inputs left blank.
    ----------
    Parameters
    ----------
    yaml_path : str or pathlib.Path
        Path to input YAML file.
    categorized_inputs_shorthands : dict
        Nested dictionary with category keys and dictionary values which each
        assign nested key shorthands for values corresponding to full parameter
        names from YAML input file.
    default_values : dict
        Shorthand keys and default values to be filled when keys are missing or
        left blank in YAML input file.
    -------
    Returns
    -------
    dict
        Dict containing inputs stored according to shorthands.
    ------
    Raises
    ------
    ValueError
        Raised if required keys (default value = 'Required') left blank.
    """
    # Open YAML file and read inputs
    stream = open(yaml_path, 'r')
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)   # User Input
    stream.close()
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
    # Change ui['out_dir_path'] to include subdirectory with name of output
    # prefix where all output files will be saved
    if Path(ui['out_dir_path']).stem != ui['out_prefix']:
        ui['out_dir_path'] = str(Path(ui['out_dir_path']) / ui['out_prefix'])
    if not Path(ui['out_dir_path']).is_dir():
        Path(ui['out_dir_path']).mkdir(parents=True)
    else:
        try:
            if not ui['overwrite']:
                raise ValueError(
                    'Output directory already exists:'
                    f" {Path(ui['out_dir_path']).resolve()}"
                )
        except KeyError:
            raise ValueError(
                'Output directory already exists:'
                f" {Path(ui['out_dir_path']).resolve()}"
            )
    # Save copy of YAML input file to output dir
    with open(
        Path(ui['out_dir_path']) / f"{ui['out_prefix']}_input.yml", 'w'
    ) as file:
        output_yaml = yaml.dump(yaml_dict, file, sort_keys=False)
    return ui

def log(logger, msg):
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

def manual_merge(img_labeled, path_to_merge_groups_txt, logger=None):
    log(logger, 'Merging the following regions:')
    merge_groups = []
    lines = open(path_to_merge_groups_txt).readlines()
    # Create a list of lists from the lines in merge group file, and convert
    # all the strings to ints
    merge_groups = [
        list(map(int, line.rstrip('\n').split(', '))) for line in lines]
    merge_labeled = img_labeled.copy()
    for regions_to_merge in merge_groups:
        log(logger, f'    {regions_to_merge}')
        for label in regions_to_merge:
            merge_labeled[img_labeled == label] = regions_to_merge[0]
    # Number of unique values. -1 accounts for 0 label
    n_merge_regions = len(np.unique(merge_labeled)) - 1
    log(logger, f'--> {n_merge_regions} region(s) after merge.')
    return merge_labeled

def merge_segmentations(imgs_semantic, imgs_instance):
    """Create a image stack that merges the semantic segmentation
    (separated classes) with the instance segmentation (single class
    with separate instances labeled) by replacing the semantic
    segmentation voxels labeled as 2 with the instance labels.
    ----------
    Parameters
    ----------
    imgs_semantic : numpy.ndarray
        DxMxN array (D slices, M rows, N columns) representing semantic
        segmentation. Classes assumed to be labeled as 0, 1, 2 with labels of
        2 replaced by instance segmentation.
    imgs_instance : numpy.ndarray
        DxMxN array (D slices, M rows, N columns) representing instance
        segmentation.
    -------
    Returns
    -------
    numpy.ndarray
        DxMxN array representing merged segmentation.
    """
    # Create new array that will represent labeled particles and binder
    imgs_merged_seg = imgs_instance.copy()
    # Replace any pixels with value 1 with an unused value so included as binder
    imgs_merged_seg[imgs_merged_seg == 1] = (
        imgs_merged_seg.max() + 1
    )
    # Set locations where binder exist (igms_thresh == 1) to 1 in new array
    imgs_merged_seg[imgs_semantic == 1] = 1
    return imgs_merged_seg

def output_checkpoints(
    fig,
    show=False,
    save_path=None,
    fn_n=0,
    fn_n_digits=2,
    fn_suffix=''
):
    """Save or show checkpoint images.

    Parameters
    ----------
    fig : matplotlib.Figure
        Matplotlob figure to be shown and/or saved.
    show : bool, optional
        If True, figure is opened in am interactive matplotlib window,
        by default False
    save_path : None or str, optional
        Path to save figure, by default None
    fn_n : int, optional
        Number used as prefix in saving of figure, by default 0
    n_digits : int, optional
        Determines number of leading zeros to add to fig_n, by default 2
    fn_suffix : str, optional
        Filename to place after the figure number, by default ''
    """
    if save_path is not None:
        if fn_suffix == '':
            fn_sep = ''
        else:
            fn_sep = '-'
        plt.savefig(
            Path(save_path)
            / f'{str(fn_n).zfill(fn_n_digits)}{fn_sep}{fn_suffix}.png')
    if show:
        plt.show()

def preprocess(
    imgs,
    median_filter=False,
    rescale_intensity_range=None,
    rescale_float_range=None,
    print_size=False,
    logger=None,
):
    """Preprocessing steps to perform on images.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing slices of a 3D volume.
    rescale_intensity_range : None or 2-tuple, optional
        Intensity range (in percent) to clip intensity. Defaults to None.
    print_size : bool, optional
        If True, print the size of the preprocessed images in GB.
        Defaults to False.
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    -------
    Returns
    -------
    numpy.ndarray, list
        3D array of the shape imgs.shape containing binarized images; list of
        threshold values used to create binarized images
    """
    log(logger, 'Preprocessing images...')
    imgs_pre = imgs.copy()
    # Apply median filter if median_filter is True
    if median_filter:
        log(logger, f'--> Applying median filter...')
        imgs_pre = filters.median(imgs_pre)
    # Rescale intensity if intensity_range passed
    if rescale_intensity_range is not None:
        log(
            logger,
            f'--> Rescaling intensities to percentile range'
            f' [{rescale_intensity_range[0]}, {rescale_intensity_range[1]}]...'
        )
        # Calculate low & high intensities
        rescale_low = np.percentile(imgs_pre, rescale_intensity_range[0])
        rescale_high = np.percentile(imgs_pre, rescale_intensity_range[1])
        # Clip low & high intensities
        imgs_pre = np.clip(imgs_pre, rescale_low, rescale_high)
        imgs_pre = exposure.rescale_intensity(
            imgs_pre, in_range='image', out_range='uint16'
        )
    elif rescale_float_range is not None:
        # Clip low & high intensities
        imgs_pre = np.clip(
            imgs_pre, rescale_float_range[0], rescale_float_range[1]
        )
        imgs_pre = exposure.rescale_intensity(
            imgs_pre,
            in_range='image',
            out_range='dtype',
        )
    log(logger, '--> Preprocessing complete.')
    if print_size:
        log(logger, f'--> Size of array (GB): {imgs_pre.nbytes / 1E9}')
    return imgs_pre

def process_args(
    argv,
    workflow_name,
    workflow_desc,
    categorized_input_shorthands,
    default_values,
):
    # Get command-line arguments
    yaml_file = ''
    if len(argv) == 0:
        help(workflow_name, workflow_desc)
        sys.exit()
    if argv[0] == '-g':
        if len(argv) == 2:
            generate_input_file(
                argv[1],
                workflow_name,
                categorized_input_shorthands,
                default_values
            )
        else:
            raise ValueError(
                'To generate an input file, pass the path of a directory'
                ' to save the file.'
            )
        sys.exit()
    elif argv[0] == '-h':
        help(workflow_name, workflow_desc)
        sys.exit()
    elif argv[0] == "-i" and len(argv) == 2:
        yaml_file = argv[1]
    if yaml_file == '':
        raise ValueError(
            f'No input file specified.'
            f' Enter "python -m segmentflow.workflow.{workflow_name} -h"'
            f' for more help'
        )
    # Load YAML inputs into a dictionary
    ui = load_inputs(yaml_file, categorized_input_shorthands, default_values)
    return ui

def radial_filter(imgs, logger=None):
    """Apply a radial filter that normalizes the intensity radially along each
    slice of a cylindrical object. Good for removing beam hardening artifacts
    in CT images. Image will be converted to 16-bit before filter is applied.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D NumPy array representing images to which the radial filtered will be
        applied.
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    -------
    Returns
    -------
    numpy.ndarray
        3D NumPy array of radially filtered images. Will be converted to 16-bit.
    """
    log(logger, 'Converting to 16-bit...')
    imgs = util.img_as_uint(imgs)
    log(logger, '--> Calculating average image...')
    img_avg = np.mean(imgs, axis=0)
    log(logger, '--> Finding edges...')
    # Calc semantic seg threshold values and generate histogram
    # threshold = filters.threshold_minimum(img_avg)
    threshold = filters.threshold_otsu(img_avg)
    # Segment images with threshold values
    img_avg_bw = isolate_classes(img_avg, threshold, intensity_step=255)
    # Detect edges in image
    edges = feature.canny(img_avg_bw)
    log(logger, '--> Fitting circle to edges...')
    cx, cy, r = fit_circle_to_edges(edges)
    log(logger, '--> Creating radial filter...')
    img_radial = np.zeros_like(img_avg)
    for r_sub in np.arange(0, r)[::-1]:
        # Draw circle
        circ_rows, circ_cols = draw.circle_perimeter(
            cy, cx, r - r_sub, shape=img_avg.shape)
        # Get the average of the 50 largest values
        circ_avg_max = np.median(
            [
                img_avg[circ_rows, circ_cols][i]
                for i in np.argsort(-img_avg[circ_rows, circ_cols])[:50]
            ]
        )
        img_radial[circ_rows, circ_cols] = circ_avg_max
    img_radial = exposure.rescale_intensity(img_radial)
    # Apply median filter to fill gaps between imperfect concentric circles
    img_radial = filters.median(img_radial)
    log(logger, '--> Applying radial filter...')
    total_med = np.median(imgs)
    img_radial[img_radial == 0] = total_med
    imgs_rad_filt = np.stack([
        imgs[i, ...] / img_radial * total_med
        for i in range(imgs.shape[0])
    ])
    # Convert to 16-bit
    imgs_rad_filt = exposure.rescale_intensity(imgs_rad_filt)
    imgs_rad_filt = util.img_as_uint(imgs_rad_filt)
    return imgs_rad_filt

def remove_particles(imgs_semantic, min_vol):
    """
    ----------
    Parameters
    ----------
    imgs_semantic : numpy.ndarray
        DxMxN (D slices, M rows, N columns) NumPy array representing semantic
        segmentation. Assumes particles = 2, matrix/binder = 1, and
        voids = 1.
    min_vol : int
        Minimum volume of particles to be retained. All others will be set to
        value of binder/matrix
    -------
    Returns
    -------
    _type_
        DxMxN array representing semantic segmentation with particles smaller
        than min_vol removed and replaced with binder/matrix.
    """
    grains_filtered = morphology.remove_small_objects(
        imgs_semantic==2, min_size=min_vol)
    imgs_semantic_rm = imgs_semantic.copy()
    imgs_semantic_rm[imgs_semantic == 2] = 1
    imgs_semantic_rm[grains_filtered==True] = 2
    return imgs_semantic_rm

def save_as_stl_files(
    segmented_images,
    stl_dir_location,
    output_prefix,
    make_new_save_dir=True,
    suppress_save_msg=True,
    slice_crop=None,
    row_crop=None,
    col_crop=None,
    stl_overwrite=False,
    spatial_res=1.0,
    n_erosions=None,
    median_filter_voxels=True,
    voxel_step_size=1,
    return_stl_dir_path=False,
    logger=None
):
    """Iterate through particles in the regions list provided by
    skimage.measure.regionprops()
    ----------
    Parameters
    ----------
    segmented_images : numpy.ndarray
        3D DxMxN array representing D segmented images with M rows and N
        columns. Each pixel/voxel of each particle is assigned a different
        integer label to differentiate from neighboring and potentially
        connected particles. Stored in "segment_dict['integer-labels']".
    stl_dir_location : Path or str
        Path to the directory where the STL files will be saved.
    output_prefix : str
        Prefix for output files
        (and new save directory if make_new_save_dir = True)
    make_new_save_dir : bool, optional
        If True, create a new directory under stl_dir_location with name
        f'{output_filename_base}STLs'. Defaults to False.
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
    return_stl_dir_path : bool, optional
        If True, return the directory path where STLs are saved.
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    -------
    Returns
    -------
    If return_dir_path is True:
        int
            Number of STL files saved.
    ------
    Raises
    ------
    ValueError
        Raise ValueError when directory named dir_name already exists at
        location save_dir_parent_path
    """
    log(logger, 'Generating surface meshes...')
    if make_new_save_dir:
        stl_dir_location = (
            Path(stl_dir_location) / f'{output_prefix}_STLs'
        )
        if stl_dir_location.is_dir():
            if not stl_overwrite:
                log(
                    logger,
                    f'Meshes not generated. Directory already exists:'
                    f'\n{stl_dir_location.resolve()}'
                )
                return
            # Else, continue to overwrite files
        else:
            stl_dir_location.mkdir()
    props_df = pd.DataFrame(columns=[
        'particleID',
        'n_voxels',
        'n_voxels_post_erosion',
        'centroid',
        'slice_min',
        'slice_max',
        'row_min',
        'row_max',
        'col_min',
        'col_max',
        'meshed',
        'stl_x_min',
        'stl_x_max',
        'stl_y_min',
        'stl_y_max',
        'stl_z_min',
        'stl_z_max',
    ])
    if n_erosions is None:
        n_erosions = 0
    regions = measure.regionprops(segmented_images)
    n_particles = len(regions)
    n_particles_digits = len(str(n_particles))
    for region in regions:
        # Create save path
        fn = (
            f'{output_prefix}_{str(region.label).zfill(n_particles_digits)}.stl'
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
        props['n_voxels_post_erosion'] = np.nan  # Replaced in erosion loop
        props['centroid']   = centroid_xyz
        props['slice_min']  = min_slice
        props['slice_max']  = max_slice
        props['row_min']    = min_row
        props['row_max']    = max_row
        props['col_min']    = min_col
        props['col_max']    = max_col
        props['meshed']     = False
        props['stl_x_min']  = np.nan
        props['stl_x_max']  = np.nan
        props['stl_y_min']  = np.nan
        props['stl_y_max']  = np.nan
        props['stl_z_min']  = np.nan
        props['stl_z_max']  = np.nan
        props['stl_is_watertight'] = False
        props['stl_volume'] = -1
        # If particle has less than 2 voxels in each dim, do not mesh surface
        # (marching cubes limitation)
        if (
            max_slice - min_slice <= 2 + 2*n_erosions
            and max_row - min_row <= 2 + 2*n_erosions
            and max_col - min_col <= 2 + 2*n_erosions
        ):
            props['meshed'] = False
            log(
                logger,
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
                # Add number of voxels in eroded particle to props dict
                props['n_voxels_post_erosion'] = np.count_nonzero(
                    imgs_particle_padded)
            if median_filter_voxels:
                # Median filter used to smooth particle in image/voxel form
                imgs_particle_padded = filters.median(imgs_particle_padded)
            # Perform marching cubes surface meshing when array has values > 0
            try:
                # Create surface mesh and save as STL file at stl_save_path
                stl_x, stl_y, stl_z = create_surface_mesh(
                    imgs_particle_padded,
                    slice_crop=slice_crop, row_crop=row_crop, col_crop=col_crop,
                    min_slice=min_slice, min_row=min_row, min_col=min_col,
                    spatial_res=spatial_res,
                    voxel_step_size=voxel_step_size,
                    save_path=stl_save_path,
                    silence=suppress_save_msg,
                    logger=logger
                )
                props['meshed'] = True
                props['stl_x_min']  = np.min(stl_x)
                props['stl_x_max']  = np.max(stl_x)
                props['stl_y_min']  = np.min(stl_y)
                props['stl_y_max']  = np.max(stl_y)
                props['stl_z_min']  = np.min(stl_z)
                props['stl_z_max']  = np.max(stl_z)
                stl_mesh = trimesh.load(stl_save_path)
                props['stl_is_watertight'] = stl_mesh.is_watertight
                if stl_mesh.is_watertight:
                    props['stl_volume'] = stl_mesh.volume
                if not suppress_save_msg:
                    log(logger, f'STL saved: {stl_save_path}')
            except RuntimeError as error:
                props['meshed'] = False
                log(
                    logger,
                    f'Surface mesh not created for particle {region.label}. '
                    'Particle likely too small. Error: ',
                    error
                )
        props_df = pd.concat(
            [props_df, pd.DataFrame.from_records([props])], ignore_index=True
        )
    csv_fn = (f'{output_prefix}_properties.csv')
    csv_save_path = Path(stl_dir_location) / csv_fn
    props_df.to_csv(csv_save_path, index=False)
    # Count number of meshed particles
    n_saved = len(np.argwhere(props_df['meshed'].to_numpy()))
    log(logger, f'--> {n_saved} STL file(s) written!')
    if return_stl_dir_path:
        return stl_dir_location

def save_binned_particles_csv(
    save_dir_path,
    bin_edges,
    n_particles,
    output_prefix='',
    logger=None,
):
    msg = 'Saving binned particles...'
    print(msg) if logger is None else logger.info(msg)
    output_prefix = ''
    if output_prefix != '':
        output_prefix += '_'
    save_path = Path(save_dir_path) / f'{output_prefix}binned_particles.csv'
    n_particles_df = pd.DataFrame(columns=['bin min', 'bin max', 'n particles'])
    if bin_edges[0] != 0:
        n_particles_df['bin min'] = np.insert(bin_edges[0:-1], 0, 0)
        n_particles_df['bin max'] = np.insert(bin_edges[1:], 0, bin_edges[0])
        n_particles_df['n particles'] = np.insert(n_particles, 0, 0)
    else:
        n_particles_df['bin min'] = bin_edges[0:-1]
        n_particles_df['bin max'] = bin_edges[1:]
        n_particles_df['n particles'] = n_particles
    n_particles_df.to_csv(save_path)
    msg = f'--> CSV saved: {save_dir_path}'
    print(msg) if logger is None else logger.info(msg)

def save_bounding_boxes(
    merge_labeled,
    out_dir_path,
    out_prefix,
    spatial_res,
):
    region_table = measure.regionprops_table(
        merge_labeled, properties=('label', 'bbox'))
    bbox_df = pd.DataFrame(region_table)
    bbox_df.rename(columns={
        'bbox-0': 'min_row',
        'bbox-1': 'min_col',
        'bbox-2': 'max_row',
        'bbox-3': 'max_col',
    }, inplace=True)
    # bbox_df['ums_per_pixel'] = spatial_res * bbox_df.index.shape[0]
    bbox_df['min_row'] *= spatial_res
    bbox_df['min_col'] *= spatial_res
    bbox_df['max_row'] *= spatial_res
    bbox_df['max_col'] *= spatial_res
    bbox_df.to_csv(Path(out_dir_path) / f"{out_prefix}_bounding_boxes.csv")

def save_bounding_coords(
    merge_labeled,
    out_dir_path,
    out_prefix,
    smooth=False,
    spatial_res=1,
    logger=None,
    return_boundary_viz=False,
    return_smoothed_viz=False,
):
    labels = np.unique(merge_labeled)
    # Pad outer edges so find_boundaries returns coordinates along
    # image borders
    merge_labeled_padded = np.pad(merge_labeled, 1)
    subpixel_bw = segmentation.find_boundaries(
        merge_labeled_padded, mode='subpixel').astype(np.ubyte)
    # Make empty array at subpixel size to hold the boundary vis
    boundary_viz = np.zeros_like(subpixel_bw)
    if smooth and return_smoothed_viz:
        # If smooth regions set to True, make an empty array at subpixel
        # size to hold the smoothing vis. Data added during loop below.
        smooth_viz = np.zeros_like(subpixel_bw)
    # Set up infra to store coordinates in CSV files (one per region)
    bounding_loops_dir_path = (
        Path(out_dir_path)
        / f"{out_prefix}_bounding_loops")
    if not bounding_loops_dir_path.exists():
        bounding_loops_dir_path.mkdir(parents=True)
    n_digits = len(str(len(labels)))
    # Iterate through labels and find boundary, order coordinates,
    # and save
    log(logger, 'Collecting region boundaries...')
    nonzero_labels = [label for label in labels if label > 0]
    log(logger,
        f'--> Number of nonzero labels: {len(nonzero_labels)}')
    for i in nonzero_labels:
        # Isolate/binarize region label
        reg_bw = np.zeros_like(merge_labeled_padded)
        reg_bw[merge_labeled_padded == i] = 1
        # Fill holes to ensure all points are around the outer edge
        reg_bw = ndimage.binary_fill_holes(reg_bw).astype(np.ubyte)
        # Find subpixel boundaries
        subpixel_bounds = segmentation.find_boundaries(
            reg_bw, mode='subpixel').astype(np.ubyte)
        boundary_viz[subpixel_bounds == 1] = i
        # Order bounding coordinates by nearest
        coords = np.transpose(np.nonzero(subpixel_bounds))
        # Order points by nearest
        loop_list = [tuple(coords[-1])]
        not_added = list(map(tuple, coords[:-1]))
        while len(loop_list) < coords.shape[0]:
            pt = tuple(loop_list[-1])
            distances = spatial.distance_matrix([pt], not_added)[0]
            nearest_i = np.argmin(distances)
            nearest_pt = not_added[nearest_i]
            not_added.pop(nearest_i)
            loop_list.append(nearest_pt)
        loop_list.append(loop_list[0])
        if smooth and return_smoothed_viz:
            loop_list, smooth_viz = smooth_bounding_coords(
                loop_list, i, smooth_viz=smooth_viz)
        elif smooth:
            loop_list = smooth_bounding_coords(
                loop_list, i, smooth_viz=None)
        loop_arr = np.array(loop_list)
        # Save ordered bounding coordinates. The division accounts for
        # the coordinates coming from a subpixel image
        # (about twice as large, plus padding & additional element).
        scale = subpixel_bw.shape[0] / merge_labeled.shape[0]
        x = loop_arr[:, 1] / scale * spatial_res
        y = loop_arr[:, 0] / scale * spatial_res
        df = pd.DataFrame(data={'x': x, 'y': y})
        df.to_csv(
            bounding_loops_dir_path / f'{str(i).zfill(n_digits)}.csv')
    csv_list = [p for p in Path(bounding_loops_dir_path).glob('*.csv')]
    log(logger,
        f'--> {len(csv_list)} region(s) saved:'
        f' {bounding_loops_dir_path}')
    if return_boundary_viz and return_smoothed_viz:
        return boundary_viz, smooth_viz
    elif return_boundary_viz:
        return boundary_viz
    elif return_smoothed_viz:
        return smooth_viz

def save_images(
    imgs,
    save_dir,
    img_names=None,
    convert_to_16bit=False,
    overwrite=False,
    logger=None
):
    """Save images to save_dir.
    ----------
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
    overwrite : bool, optional
        If True, existing directory will be overwritten. Defaults to False.
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    """
    save_dir = Path(save_dir)
    # Create directory and raise error if dir already exists and overwrite False
    if not overwrite:
        save_dir.mkdir(parents=True, exist_ok=False)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
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
            n_imgs = len(imgs)
            # Pad front of image name with zeros to match longest number
            img_name = str(i).zfill(len(str(n_imgs)))
        else:
            img_name = img_names[i]
        iio.imwrite(Path(save_dir / f'{img_name}.{file_suffix}'), img)
    log(logger, f'{len(imgs)} image(s) saved to: {save_dir.resolve()}')

def save_isolated_classes(imgs, thresh_vals, save_dir_path):
    print('Saving isolated classes as binary images...')
    save_dir_path = Path(save_dir_path)
    if not save_dir_path.is_dir():
        save_dir_path.mkdir()
    # If class_idxs is not a list, make it a single item list
    if not isinstance(thresh_vals, np.ndarray):
        thresh_vals = [thresh_vals]
    # Sort thresh_vals in ascending order then reverse to get largest first
    thresh_vals.sort()
    isolated_classes = np.zeros_like(imgs, dtype=np.uint8)
    # Starting with the lowest threshold value, set pixels above each
    # increasing threshold value to an increasing unique marker (1, 2, etc.)
    for i, val in enumerate(thresh_vals):
        isolated_classes[imgs > val] = i + 1
    # Save isolated_classes
    classes_save_dir = Path(save_dir_path) / 'isolated-classes'
    if not classes_save_dir.is_dir():
        classes_save_dir.mkdir()
    n_digits = len(str(isolated_classes.shape[0]))
    for img_i in range(isolated_classes.shape[0]):
        save_path = (
                Path(classes_save_dir)
                / f'isolated-classes_{str(img_i).zfill(n_digits)}.tiff')
        iio.imwrite(save_path, isolated_classes[img_i, ...])
    print(f'{len(imgs)} image(s) saved to: {classes_save_dir.resolve()}')

def save_properties_csv(
        imgs_labeled,
        output_prefix,
        save_dir_path,
        return_save_dir_path=False
):
    props_df = pd.DataFrame(columns=[
        'particleID',
        'n_voxels',
        'centroid',
        'min_slice',
        'max_slice',
        'min_row',
        'max_row',
        'min_col',
        'max_col'
    ])
    regions = measure.regionprops(imgs_labeled)
    # n_particles = len(regions)
    for region in regions:
        # Get bounding slice, row, and column
        min_slice, min_row, min_col, max_slice, max_row, max_col = region.bbox
        # Get centroid coords in slice, row, col and reverse to get x, y, z
        centroid_x, centroid_y, centroid_z = (
            reversed([str(round(coord)) for coord in region.centroid])
        )
        props = {}
        props['particleID'] = region.label
        props['n_voxels']   = region.area
        props['centroid_x'] = centroid_x
        props['centroid_y'] = centroid_y
        props['centroid_z'] = centroid_z
        props['min_slice']  = min_slice
        props['max_slice']  = max_slice
        props['min_row']    = min_row
        props['max_row']    = max_row
        props['min_col']    = min_col
        props['max_col']    = max_col
        props_df = pd.concat(
            [props_df, pd.DataFrame.from_records([props])], ignore_index=True
        )
    csv_fn = (f'{output_prefix}_properties.csv')
    csv_save_path = Path(save_dir_path) / csv_fn
    props_df.to_csv(csv_save_path, index=False)
    if return_save_dir_path:
        return save_dir_path

def save_shell_vertices(
        img_dir_path,
        save_dir_path,
        slice_crop=None,
        row_crop=None,
        col_crop=None,
        slice_offset=0,
        scale=1,
        file_suffix='.tif'
    ):
    img_dir_path = Path(img_dir_path)
    save_dir_path = Path(save_dir_path)
    save_path = (
        save_dir_path / f'{img_dir_path.name[:]}'
        f'/{img_dir_path.name}_shell_{scale}scale_slices-'
        f'{str(slice_crop[0]).zfill(4)}-{str(slice_crop[1]).zfill(4)}.csv'
    )
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    imgs = load_images(
        img_dir_path,
        slice_crop=slice_crop,
        row_crop=row_crop,
        col_crop=col_crop,
        file_suffix=file_suffix
    )
    # Downscale iamges
    if scale != 1:
        print('Downsizing images...')
        imgs = transform.rescale(imgs, scale, anti_aliasing=False)
        print('--> Images downsized:', imgs.shape)
    # Plot intensity rescale histogram
    imgs = preprocess(
        imgs, median_filter=True,
        rescale_intensity_range=[0.01, 99.99])
    imgs = util.img_as_uint(imgs)
    # Calc semantic seg threshold values and generate histogram
    thresholds = threshold_multi_otsu(
        imgs, nclasses=2, nbins=256, convert_to_float=False
    )
    # Segment images
    imgs = isolate_classes(imgs, thresholds)
    # Create shell
    imgs_shell = imgs - morphology.binary_erosion(imgs)
    shell_verts = np.argwhere(imgs_shell == 1)
    shell_df = pd.DataFrame(shell_verts, columns=['slice', 'row', 'column'])
    shell_df['slice'] = shell_df['slice'] + slice_offset
    shell_df['row'] = shell_df['row'] + int(round(row_crop[0] * scale))
    shell_df['column'] = shell_df['column'] + int(round(col_crop[0] * scale))
    print('Total n voxels:', math.prod(i for i in imgs.shape))
    print('Shell n vertices:', shell_df.index.shape[0])
    save_path = (
        save_dir_path / f'{img_dir_path.name[:]}'
        f'/{img_dir_path.name}_shell_{scale}scale_slices-'
        f'{str(slice_crop[0]).zfill(4)}-{str(slice_crop[1]).zfill(4)}.csv'
    )
    shell_df.to_csv(save_path, index=False)
    if save_path.exists():
        print(f'Shell vertices saved to CSV: {save_path}')

def save_vtk(
        img_dir_path,
        save_path,
        file_suffix='.tif',
        convert_to_16bit=False,
        overwrite=False
    ):
    img_dir_path = Path(img_dir_path)
    save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    if save_path.exists() and not overwrite:
        raise ValueError('File already exists:', save_path)
    # Create 3D array for storing tiff series
    imgs = load_images(img_dir_path, file_suffix=file_suffix)
    if convert_to_16bit:
        imgs = util.img_as_uint(imgs)
    n_slices, n_rows, n_cols = imgs.shape
    # Write the Paraview File
    print('Saving VTK file...')
    # Open file
    with open(save_path, 'w') as f:
        # Write metadata
        n_pts = n_slices * n_rows * n_cols
        header = [
            '# vtk DataFile Version 2.0\n',
            f'{img_dir_path.name}\n',
            'ASCII\n',
            'DATASET STRUCTURED_POINTS\n',
            f'DIMENSIONS {n_cols} {n_rows} {n_slices}\n',
            'ASPECT_RATIO 1 1 1\n',
            'ORIGIN 0 0 0\n',
            '\n',
            f'POINT_DATA {n_pts}\n',
            'SCALARS Brightness float\n',
            'LOOKUP_TABLE default\n',
        ]
        f.writelines(header)
        for i in range(n_slices):
            print(
                f'--> Writing values in slice '
                f'{str(i+1).zfill(len(str(n_slices)))}/{n_slices}...'
            )
            for j in range(n_rows):
                for k in range(n_cols):
                    f.write(f'{imgs[i, j, k]}\n')
    print('VTK file saved:', save_path)

def simulate_sieve_bbox(dims_df, bin_edges, pixel_res, logger=None):
    """Simulate sieve using the Cartesian bounding box of each particle.
    ----------
    Parameters
    ----------
    dims_df : pandas.DataFrame
        DataFrame object with the columns "nslices", "nrows", and "ncols"
    bin_edges : numpy.array, list, or str
        Particle diameter sizes denoting the bin edges of the size distribution.
        Can also pass "F50" to use the standard bin edges for F50 sand, or
        "IDOX" to use the standard bin edges for IDOX.
    pixel_res : float
        Size of voxel in same units as bin_edges. Assumes cubic voxels.
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    -------
    Returns
    -------
    numpy.array, numpy.array
        Two 1D numpy arrays denoting the number of particles in each bin
        (size: N) and the sieve size/bin edges (size: N + 1) when N is the
        number of bins.
    """
    msg = 'Simulating sieve based on bounding box aspect ratios...'
    print(msg) if logger is None else logger.info(msg)
    if isinstance(bin_edges, str):
        if bin_edges.lower() == 'f50':
            msg = '--> Bins set based on expected distribution of F50 sand'
            print(msg) if logger is None else logger.info(msg)
            bin_edges = [53,  75, 106, 150, 212, 300,  425, 600, 850]
        elif bin_edges.lower() == 'idox':
            msg = '--> Bins set based on expected distribution of IDOX'
            print(msg) if logger is None else logger.info(msg)
            bin_edges = [0, 45, 75, 150, 300]
        else:
            raise ValueError(
                'Only "F50" or "IDOX" can be passed as a string.')
    # Define dimensions a, b, c with a as largest and c as smallest
    dims_df['a'] = dims_df.apply(
        lambda row: row['nslices' : 'ncols'].astype(int).nlargest(3).iloc[0],
        axis=1
    )
    dims_df['b'] = dims_df.apply(
        lambda row: row['nslices' : 'ncols'].astype(int).nlargest(3).iloc[1],
        axis=1
    )
    dims_df['c'] = dims_df.apply(
        lambda row: row['nslices' : 'ncols'].astype(int).nlargest(3).iloc[2],
        axis=1
    )
    # Apply pixel resolution to second smallest dimension
    b_ums = pixel_res * dims_df['b'].to_numpy()
    n_particles, sieve_sizes = np.histogram(b_ums, bins=bin_edges)
    return n_particles, sieve_sizes

def smooth_bounding_coords(
    loop_list,
    label,
    smooth_viz=None,
):
    smooth_list = []
    # Special case of the for loop below where pt before is
    # actually the second to last in the list, since the first
    # pt is repeated at the end of the list to make it a closed
    # loop
    if (
        spatial.distance.euclidean(loop_list[-2], loop_list[1])
        >= (
            spatial.distance.euclidean(
                loop_list[0], loop_list[-2])
            + spatial.distance.euclidean(
                loop_list[0], loop_list[1])
        )
    ):
        smooth_list.append(loop_list[0])
    for pt_i in range(1, len(loop_list) - 1):
        # If distance between pt before & pt after is larger or
        # equal to the the sum of the distance between the
        # current pt & the pt before with the distance between
        # the current pt & the pt after, copy pt to smooth_list.
        # This means pts farther away than the surrounding pts
        # are removed.
        if (
            spatial.distance.euclidean(
                loop_list[pt_i - 1], loop_list[pt_i + 1])
            >= (
                spatial.distance.euclidean(
                    loop_list[pt_i], loop_list[pt_i - 1])
                + spatial.distance.euclidean(
                    loop_list[pt_i], loop_list[pt_i + 1])
            )
        ):
            smooth_list.append(loop_list[pt_i])
    # Add first pt in list to the end to make it a closed loop
    smooth_list.append(smooth_list[0])
    loop_list = smooth_list
    smooth_coords = np.array(smooth_list)
    if smooth_viz is not None:
        smooth_viz[smooth_coords[:, 0], smooth_coords[:, 1]] = label
        return smooth_list, smooth_viz
    else:
        return smooth_list

def threshold_multi_min(
    imgs,
    nbins=256,
    nthresholds='all',
    return_fig_ax=False,
    ylims=None,
    signal_kwargs={},
    plt_kwargs={},
):
    """Semantic segmentation by detecting multiple minima in the histogram.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D NumPy array representing images for which thresholds will be
        determined.
    nbins : int
        Number of bins used to calculate histogram.
    signal_kwargs : dict, optional
        Passed to scipy.signal.find_peaks() when calculating maxima.
    -------
    Returns
    -------
    list
        List of intensity minima that can be used to threshold the image.
        Values will be 16-bit if imgs passed is 16-bit, else float.
    """
    print('Calculating thresholds from local minima...')
    # if imgs.dtype == 'uint16':
    #     imgs = util.img_as_float32(imgs)
    # else:
    #     raise ValueError(
    #         'Input images must be converted to 16-bit before continuing.')
    # Calculate histogram
    hist, hist_centers = exposure.histogram(imgs, nbins=nbins)
    # Smooth histogram with Gaussian filter
    hist_smooth = scipy.ndimage.gaussian_filter(hist, 3)
    # Find local maxima in smoothed histogram
    peaks, peak_props = scipy.signal.find_peaks(hist_smooth, **signal_kwargs)
    # peaks_adjusted = [int(hist_centers[i] * 65536) for i in peaks]
    peaks_adjusted = [hist_centers[i] for i in peaks]
    print(f'--> {len(peaks)} peak(s) found: {peaks_adjusted}')
    # Find minima between each neighboring pair of local maxima
    min_inds = []
    for i in range(1, len(peaks)):
        min_sub_i = np.argmin(hist_smooth[peaks[i - 1] : peaks[i]])
        min_inds.append(min_sub_i + peaks[i - 1])
    min_counts = [hist[i] for i in min_inds]
    # mins = [int(hist_centers[i] * 65536) for i in min_inds]
    mins = [hist_centers[i] for i in min_inds]
    # Create dictionary with number of counts (keys) for each minima (values)
    mins_by_counts = {k : v for k, v in zip(min_counts, mins)}
    # Sort disctionary according to number of counts (low to high)
    mins_by_counts = {
        k : mins_by_counts[k] for k in sorted(mins_by_counts.keys())}
    # Select first nthreshold minima
    if nthresholds != 'all':
        mins = list(mins_by_counts.values())[:nthresholds]
    print(f'--> {len(mins)} minima found: {mins}')
    if return_fig_ax:
        # Plot peaks & mins on histograms
        fig, ax = plt.subplots(facecolor='white', **plt_kwargs)
        # ax.plot(hist_centers * 65536, hist, label='Histogram')
        # ax.plot(hist_centers * 65536, hist_smooth, c='C1', label='Smoothed')
        ax.plot(hist_centers, hist, label='Histogram')
        ax.plot(hist_centers, hist_smooth, c='C1', label='Smoothed')
        if ylims is not None:
            ax.set_ylim(ylims)
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            x=peaks_adjusted, ymin=ymin, ymax=ymax, colors='C3', label='Maxima')
        ax.vlines(x=mins, ymin=ymin, ymax=ymax, colors='C2', label='Thresholds')
        ax.legend()
        return mins, fig, ax
    else:
        return mins

def threshold_multi_otsu(
    imgs,
    nclasses=2,
    return_fig_ax=False,
    ylims=None,
    convert_to_float=True,
    **kwargs
):
    """Semantic segmentation by application of the Multi Otsu algorithm.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D NumPy array representing images for which thresholds will be
        determined.
    nclasses : int
        Number of classes to  used to calculate histogram.
    convert_to_float : bool
        If True, convert images to floating point values before determining
        thresholds. Defaults to True.
    -------
    Returns
    -------
    list
        List of intensity minima that can be used to threshold the image.
        Values will be 16-bit if imgs passed is 16-bit, else float.
    """
    print('Calculating Multi Otsu thresholds...')
    # if imgs.dtype == 'uint16':
    #     imgs = util.img_as_float32(imgs)
    # else:
    #     raise ValueError(
    #         'Input images must be converted to 16-bit before continuing.')
    imgs_flat = imgs.flatten()
    thresh_vals = filters.threshold_multiotsu(imgs_flat, nclasses)
    # Calculate histogram
    hist, hist_centers = exposure.histogram(imgs, nbins=256)
    if return_fig_ax:
        # Plot peaks & mins on histograms
        fig, ax = plt.subplots()
        # if convert_to_float:
        #     ax.plot(hist_centers * 65536, hist, label='Histogram')
        ax.plot(hist_centers, hist, label='Histogram')
        if ylims is not None:
            ax.set_ylim(ylims)
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            x=thresh_vals, ymin=ymin, ymax=ymax, colors='C2',
            label='Thresholds'
        )
        ax.legend()
        return thresh_vals, fig, ax
    else:
        return thresh_vals

def watershed_segment(
    imgs_binarized,
    min_peak_distance=1,
    use_int_dist_map=False,
    exclude_borders=False,
    print_size=False,
    return_dict=False,
    logger=None,
):
    """Create images with regions segmented and labeled using a watershed
    segmentation algorithm.
    ----------
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
    logger : logging.Logger, optional
        If not None, print statements will also be passed to a file determined
        at the creation of the Logger.
        See segmentflow.workflows.Workflow.create_logger. Defaults to None.
    -------
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
    if logger is None:
        print('Segmenting images...')
    else:
        logger.info('Segmenting images...')
    dist_map = ndimage.distance_transform_edt(imgs_binarized)
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
        if logger is None:
            print(f'Calculated min_peak_distance: {min_peak_distance}')
        else:
            logger.info(f'Calculated min_peak_distance: {min_peak_distance}')
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
        -1 * dist_map.astype(np.int32), seeds.astype(np.int32),
        mask=imgs_binarized.astype(np.int32)
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
        if logger is None:
            print('Excluding borders...')
        else:
            logger.info('Excluding borders...')
        if logger is None:
            print(
                '--> Number of particle(s) before border exclusion: ',
                str(n_particles))
        else:
            logger.info(
                '--> Number of particle(s) before border exclusion: ',
                str(n_particles))
        labels = segmentation.clear_border(labels)
        # Calculate number of instances of each value in label_array
        particleIDs = np.unique(labels)
        # Subtract 1 to account for background label
        n_particles = len(particleIDs) - 1
    if logger is None:
        print(
            f'--> Segmentation complete. '
            f'{n_particles} particle(s) segmented.')
    else:
        logger.info(
            f'--> Segmentation complete. '
            f'{n_particles} particle(s) segmented.')
    if print_size:
        # sys.getsizeof() doesn't represent nested objects; need to add manually
        if logger is None:
            print('--> Size of segmentation results (GB):')
        else:
            logger.info('--> Size of segmentation results (GB):')
        for key, val in segment_dict.items():
            if logger is None:
                print(f'----> {key}: {sys.getsizeof(val) / 1E9}')
            else:
                logger.info(f'----> {key}: {sys.getsizeof(val) / 1E9}')
    if return_dict:
        segment_dict = {
            'distance-map' : dist_map,
            'maxima' : maxima,
            'integer-labels' : labels,
        }
        return segment_dict
    else:
        return labels

