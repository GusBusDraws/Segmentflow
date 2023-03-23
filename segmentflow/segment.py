#!/usr/bin/python3
#Requires: Python version >= 3.5
#          PyYAML (import yaml)

#~~~~~~~~~~#
# Packages #
#~~~~~~~~~~#
import getopt
import imageio.v3 as iio
import numpy as np
from pathlib import Path
import pandas as pd
import scipy
import scipy.ndimage as ndi
from skimage import (
        exposure, feature, filters, morphology, measure,
        segmentation, util )
from stl import mesh
import sys


#~~~~~~~~~~~#
# Functions #
#~~~~~~~~~~~#
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
        Defaults to 'tif'
    print_size : bool, optional
        If True, print size of loaded images in GB. Defaults to False.
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
    print('Loading images...')
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
    -------
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

def multi_min_threshold(imgs, nbins=256, **kwargs):
    """Semantic segmentation by detecting multiple minima in the histogram.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D NumPy array representing images for which thresholds will be
        determined.
    nbins : int
        Number of bins used to calculate histogram.
    kwargs : various, optional
        Passed to scipy.signal.find_peaks() when calculating maxima.
    -------
    Returns
    -------
    list
        List of intensity minima that can be used to threshold the image.
        Values will be 16-bit if imgs passed is 16-bit, else float.
    """
    print('Calculating thresholds from local minima...')
    originally_16bit = False
    if imgs.dtype == np.uint16:
        originally_16bit = True
    if imgs.dtype != float:
        imgs = util.img_as_float32(imgs)
    # Calculate histogram
    hist, hist_centers = exposure.histogram(imgs, nbins=nbins)
    # Smooth histogram with Gaussian filter
    hist_smooth = scipy.ndimage.gaussian_filter(hist, 3)
    # Find local maxima in smoothed histogram
    peaks, peak_props = scipy.signal.find_peaks(hist_smooth, **kwargs)
    if originally_16bit:
        peaks_adjusted = [int(hist_centers[i] * 65536) for i in peaks]
    else:
        peaks_adjusted = [hist_centers[i] for i in peaks]
    print(f'--> {len(peaks)} peak(s) found: {peaks_adjusted}')
    # Find minima between each neighboring pair of local maxima
    mins = []
    for i in range(1, len(peaks)):
        min_sub_i = np.argmin(hist_smooth[peaks[i - 1] : peaks[i]])
        mins.append(min_sub_i + peaks[i - 1])
    # Convert minima indices to intensity values (16-bit or float)
    if originally_16bit:
        mins = [int(hist_centers[i] * 65536) for i in mins]
    else:
        mins = [hist_centers[i] for i in mins]
    print(f'--> {len(mins)} minima found: {mins}')
    return mins

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

def isolate_classes(
    imgs,
    threshold_values,
    intensity_step=1,
):
    """Threshold array with multiple threshold values.
    ----------
    Parameters
    ----------
    imgs : list
        3D NumPy array or list of 2D arrays representing images to be plotted.
    threshold_values : list or float, optional
        Float or list of floats to segment image.
    intensity_step : int, optional
        Step value separating intensities. Defaults to 1, but might be set to
        soemthing like 125 such that isolated classes could be viewable in
        saved images.
    -------
    Returns
    -------
    matplotlib.Figure, matplotlib.Axis
        2-tuple containing matplotlib figure and axes objects
    """
    # Sort thresh_vals in ascending order then reverse to get largest first
    threshold_values.sort()
    imgs_thresh = np.zeros_like(imgs, dtype=np.uint8)
    # Starting with the lowest threshold value, set pixels above each
    # increasing threshold value to an increasing unique marker (1, 2, etc.)
    # multiplied by the intesnity_step parameter
    for i, val in enumerate(threshold_values):
        imgs_thresh[imgs > val] = int((i + 1) * intensity_step)
    return imgs_thresh

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
    ----------
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
    -------
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
        silence=False
):
    if not silence:
        print('Creating surface mesh with marching cubes algorithm...')
    verts, faces, normals, values = measure.marching_cubes(
        imgs, step_size=voxel_step_size,
        allow_degenerate=False
    )
    if not silence:
        print('Converting mesh to STL format...')
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
    return_n_saved : bool, optional
        If True, the number of particles saved will be returned.
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
    print('Generating surface meshes...')
    if make_new_save_dir:
        stl_dir_location = (
            Path(stl_dir_location) / f'{output_prefix}_STLs'
        )
        if stl_dir_location.is_dir():
            print(
                f'Meshes not generated. Directory already exists:'
                f'\n{stl_dir_location.resolve()}'
            )
            return
        else:
            stl_dir_location.mkdir()
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
            f'{output_prefix}_'
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
                # Create surface mesh and save as STL file at stl_save_path
                vertices, faces, normals, vals = create_surface_mesh(
                    imgs_particle_padded, slice_crop=slice_crop,
                    row_crop=row_crop, col_crop=col_crop,
                    min_slice=min_slice, min_row=min_row, min_col=min_col,
                    spatial_res=spatial_res,
                    voxel_step_size=voxel_step_size,
                    save_path=stl_save_path,
                    silence=suppress_save_msg
                )
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
    csv_fn = (f'{output_prefix}_properties.csv')
    csv_save_path = Path(stl_dir_location) / csv_fn
    props_df.to_csv(csv_save_path, index=False)
    # Count number of meshed particles
    n_saved = len(np.argwhere(props_df['meshed'].to_numpy()))
    print(f'--> {n_saved} STL file(s) written!')

def save_images(
    imgs,
    save_dir,
    img_names=None,
    convert_to_16bit=False
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
            n_imgs = len(imgs)
            # Pad front of image name with zeros to match longest number
            img_name = str(i).zfill(len(str(n_imgs)))
        else:
            img_name = img_names[i]
        iio.imsave(Path(save_dir / f'{img_name}.{file_suffix}'), img)
    print(f'{len(imgs)} image(s) saved to: {save_dir.resolve()}')

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

