#!/usr/bin/python3

#~~~~~~~~~
# Packages
#~~~~~~~~~

import getopt
import imageio as iio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from pathlib import Path
from scipy import ndimage as ndi
from skimage import color, exposure, feature, filters, morphology, measure, segmentation, util
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
    print('||  Sorry, there has been a fatal error. Error message follows this banner.')
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
    print('||  This script converts CT scans of samples containing particles to STL files, where')
    print('||  each STL file describes one of the particles in the sample.')
    print('||')
    print('==')
    print()
    print('Usage')
    print()
    print('   ./segment.py -f <inputFile.yml>')
    print()
    print('where <inputFile.yml> is the path to your YAML input file.  See the example input file')
    print('in the repo top-level directory to learn more about the content (inputs) of the input file.')
    print()
    exit(0)

#~~~~~~~~~~
# Functions
#~~~~~~~~~~

def load_images(
    img_dir,
    slice_crop=None, 
    row_crop=None, 
    col_crop=None, 
    also_return_names=False,
    also_return_dir_name=False,
    convert_to_float=False,
    file_suffix='tif'
):
    """Load images from path and return as list of 2D arrays. Can also return names of images.

    Parameters
    ----------
    img_dir : str or Path
        Path to directory containing images to be loaded.
    slice_crop : list or None
        Cropping limits of slice dimension (imgs.shape[0]) of 3D array of images. Essentially chooses subset of images from sorted image directory. If None, all images/slices will be loaded. Defaults to None.
    row_crop : str or None
        Cropping limits of row dimension (imgs.shape[1]) of 3D array of images. If None, all rows will be loaded. Defaults to None.
    col_crop : str or None
        Cropping limits of column dimension (imgs.shape[2]) of 3D array of images. If None, all columns will be loaded. Defaults to None.
    also_return_names : bool, optional
        If True, returns a list of the names of the images in addition to the list of images themselves. Defaults to False.
    also_return_dir_name : bool, optional
        If True, returns a string representing the name of the image directory in addition to the list of images themselves. Defaults to False.
    convert_to_float : bool, optional
        If True, convert loaded images to floating point images, else retain their original dtype. Defaults to False
    file_suffix : str, optional
        File suffix of images that will be loaded from img_dir. Defaults to 'tif'

    Returns
    -------
    list, numpy.ndarray, or tuple
        List of arrays or 3D array representing images (depending on return_3d_array), or if also_return_names is True, list containing names of images from filenames is also returned.
    """
    img_path_list = [
        path for path in Path(img_dir).glob(f'*{file_suffix}')
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
        imgs[i, ...] = iio.imread(img_path)[row_crop[0]:row_crop[1], col_crop[0]:col_crop[1]]
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
        Value to threshold point images. Defaults to 0.65 for floating point images.
    fill_holes : str or int, optional
        If 'all', all holes will be filled, else if integer, all holes with an area in pixels below that value will be filled in binary array/images. Defaults to 64.
    return_process_dict : bool, optional
        If True, return a dictionary containing all processing steps instead of last step only, defaults to False

    Returns
    -------
    numpy.ndarray or dict
        If return_process_dict is False, a 3D array representing the hole-filled, binary images, else a dictionary is returned with a 3D array for each step in the binarization process.
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
):
    """Preprocessing steps to perform on images.

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing slices of a 3D volume.
    rescale_intensity_range : None or 2-tuple, optional
        Intensity range (in percent) to clip intensity.

    Returns
    -------
    numpy.ndarray, list
        3D array of the shape imgs.shape containing binarized images; list of threshold values used to create binarized images 
    """
    imgs_pre = imgs.copy()
    # Apply median filter if median_filter is True
    if median_filter:
        print(f'Applying median filter...')
        imgs_pre = filters.median(imgs_pre)
    # Rescale intensity if intensity_range passed
    if rescale_intensity_range is not None:
        print(f'Rescaling intensities to percentile range {rescale_intensity_range}...')
        # Calculate low & high intensities
        rescale_low = np.percentile(imgs_pre, rescale_intensity_range[0])
        rescale_high = np.percentile(imgs_pre, rescale_intensity_range[1])
        # Clip low & high intensities
        imgs_pre = np.clip(imgs_pre, rescale_low, rescale_high)
        imgs_pre = exposure.rescale_intensity(
            imgs_pre, in_range='image', out_range='uint16'
        )
    return imgs_pre

def binarize_multiotsu(
    imgs, 
    n_otsu_classes=2, 
    n_selected_thresholds=1, 
    exclude_borders=False,
):
    """Binarize stack of images (3D array) using multi-Otsu thresholding algorithm.

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array representing slices of a 3D volume.
    n_otsu_classes : int, optional
        Number of classes to threshold images, by default 2
    n_selected_thresholds : int, optional
        Number of classes to group together (from the back of the thresholded values array returned by multi-Otsu function) to create binary image, by default 1
    exclude_borders : bool, optional
        If True, exclude particles that touch the border of the volume chunk specified by slice/row/col crop in load_images(). Defaults to False.

    Returns
    -------
    numpy.ndarray, list
        3D array of the shape imgs.shape containing binarized images; list of threshold values used to create binarized images 
    """
    imgs_binarized = np.zeros_like(imgs, dtype=np.uint8)
    print('Calculating Otsu threshold(s)...')
    imgs_flat = imgs.flatten()
    thresh_vals = filters.threshold_multiotsu(imgs_flat, n_otsu_classes)
    # In an 8-bit image (uint8), the max value is 255
    # The top regions are selected by counting backwards (-) with 
    # n_selected_thresholds
    imgs_binarized[imgs > thresh_vals[-n_selected_thresholds]] = 255
    # Remove regions of binary image at borders of array
    if exclude_borders:
        imgs_binarized = segmentation.clear_border(imgs_binarized)
    return imgs_binarized, thresh_vals

def watershed_segment(
    imgs_binarized, 
    min_peak_distance=1,
    use_int_dist_map=False,
    return_dict=False
):
    """Create images with regions segmented and labeled using a watershed segmentation algorithm.

    Parameters
    ----------
    binarized_imgs : numpy.ndarray
        3D DxMxN array representing D binary images with M rows and N columns to be used in segmentation.
    min_peak_distance : int or str, optional
        Minimum distance (in pixels) of local maxima to be used to generate seeds for watershed segmentation algorithm. 'median' can be passed to use the radius of the circle with equivalent area to the median binary region. Defaults to 1.
    use_int_dist_map : bool, optional
        If true, convert distance map to 16-bit array. Use with caution-- changes segmentation results
    return_dict : bool, optional
        If true, return dict, else return 3D array with pixels labeled corresponding to unique particle integers (see below)

    Returns
    -------
    if return_dict == True :
        dict
            Dictionary of 3D DxMxN arrays the segmentation steps and labeled images. Keys for dict: 'binarized', 'distance-map', 'maxima-points', 'maxima-mask', 'seeds', 'integer-labels'
    if return_dict == False :
        numpy.ndarray
            3D DxMxN array representing segmented images with pixels labeled corresponding to unique particle integers
    """
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
        Dictionary containing segmentation routine steps, as returned from watershed_segment(). Must contain key 'integer-labels'
    particleID : int or None, optional
        If an integer is passed, only the number of voxels within the particle matching that integer are returned.
    exclude_zero : bool, optional
        Exclude zero label in count. Usually zero refers to background. Defaults to True

    Returns
    -------
    If particleID is not None:
        int
            Number of voxels in particle labeled as particleID in segmment_dict['integer-labels']
    Else:
        dict
            Dictionary with particleID keys and integer number of voxels in particle corresponding to particleID key.
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
        Dictionary containing segmentation routine steps, as returned from watershed_segment(), with at least the key: 'integer-labels' and corresponding value: images with segmented particles labeled with unique integers
    particleID : int
        Label corresponding to pixels in segment_dict['integer-labels'] that will be plotted 
    erode : bool, optional
        If True, isolated particle will be eroded before array is returned.

    Returns
    -------
    numpy.ndarray
        3D array of the same size as segment_dict['integer-labels'] that is only nonzero where pixels matched value of integer_label in original array
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
        Path at which STL file will be saved. If doesn't end with '.stl', it will be added.
    verts : array-like
        Array of (x, y, z) vertices indexed with faces to construct triangles.
    faces : array-like
        Array of indices referencing verts that define the triangular faces of the mesh.
    spatial_res : float, optional
        Factor to apply to multiply spatial vectors of saved STL. Applying the spatial/pixel resolution of the CT scan will give the STL file units of the value. Defaults to 1 to save the STL in units of pixels.
    x_offset : int, optional
        Integer value to offset x coordinates of STL. Related to column crop and particel position.
    y_offset : int, optional
        Integer value to offset y coordinates of STL. Related to row crop and particel position.
    z_offset : int, optional
        Integer value to offset z coordinates of STL. Related to slice crop and particel position.
    suppress_save_message : bool, optional
        If True, particle label and STL file path will not be printed. By default False
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

def save_regions_as_stl_files(
    imgs,
    regions,
    stl_dir_location,
    output_filename_base,
    n_particles_digits,
    suppress_save_msg=True,
    slice_crop=None,
    row_crop=None,
    col_crop=None,
    spatial_res=1,
    voxel_step_size=1,
    erode_particles=False,
    stl_overwrite=False,
    return_n_saved=True,
):
    """Iterate through particles in the regions list provided by skimage.measure.regionprops()

    Parameters
    ----------
    imgs : numpy.ndarray
        3D array of full volume.
    regions : list of skimage.RegionProperties
        List of regions that will be iterated across to position the particle in the full array.
    stl_dir_location : Path or str
        Path to the directory where the STL files will be saved.
    n_particles_digits : int
        Number of digits to denote particle label. Determines number of leading zeros.
    suppress_save_msg : bool, optional
        If True, save messages are not printed for each STL. Defaults to True.
    slice_crop : list or None, optional
        Min and max crop in the slice dimension.
    row_crop : list or None, optional
        Min and max crop in the row dimension.
    col_crop : list or None, optional
        Min and max crop in the column dimension.
    spatial_res : float, optional
        Factor to apply to multiply spatial vectors of saved STL. Applying the spatial/pixel resolution of the CT scan will give the STL file units of the value. Defaults to 1 to save the STL in units of pixels.
    voxel_step_size : int, optional
        Number of voxels to iterate across in marching cubes algorithm. Larger steps yield faster but coarser results. Defaults to 1. 
    erode_particles : bool, optional
        If True, morphologic erosion performed to remove one layer of voxels from outer layer of particle. Defaults to False.
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
        Raise ValueError when directory named dir_name already exists at location save_dir_parent_path
    """
    n_saved = 0
    for region in regions:
        # 3D area is actually volume (N voxels)
        n_voxels = region.area
        # Get bounding slice, row, and column
        min_slice, min_row, min_col, max_slice, max_row, max_col = region.bbox
        # Continue with process if particle has at least 2 voxels in each dim
        if (
            max_slice - min_slice >= 2 
            and max_row - min_row >= 2 
            and max_col - min_col >= 2
        ):
            # Isolate Individual Particles
            imgs_particle = region.image
            if erode_particles:
                imgs_particle = morphology.binary_erosion(imgs_particle)
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
            # Pad imgs_particle in imgs_particle_full to accomodate offset
            imgs_particle_full = np.zeros(
                (imgs.shape[0] + z_offset, imgs.shape[1] + y_offset, imgs.shape[2] + x_offset), 
                dtype=np.uint8
            )
            imgs_particle_full[
                z_offset + min_slice : z_offset + max_slice, 
                y_offset + min_row : y_offset + max_row, 
                x_offset + min_col : x_offset + max_col
            ] = imgs_particle
            # Do Surface Meshing - Marching Cubes
            verts, faces, normals, values = measure.marching_cubes(
                imgs_particle_full, step_size=voxel_step_size
            )
            # Create save path
            fn = (
                f'{output_filename_base}'
                f'{str(region.label).zfill(n_particles_digits)}.stl'
            )
            stl_save_path = Path(stl_dir_location) / fn
            # Save STL
            if stl_overwrite and stl_save_path.exists():
                stl_save_path.unlink()
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
                # Write the mesh to STL file
                stl_mesh.save(save_path)
                n_saved += 1
                if not suppress_save_msg:
                    print(f'STL saved: {save_path}')
    if return_n_saved:
        return n_saved

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
        Images to save, either as a list or a 3D numpy array (4D array of colored images also works)
    save_dir : str or Path
        Path to new directory to which iamges will be saved. Directory must not already exist to avoid accidental overwriting. 
    img_names : list, optional
        List of strings to be used as image filenames when saved. If not included, images will be names by index. Defaults to None.
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
        Array of indices referencing verts that define the triangular faces of the mesh.

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

def plot_stl(stl_path, zoom=True):
    """Load an STL and plot it using matplotlib.

    Parameters
    ----------
    stl_path : Path or str
        Path to an STL file to load or a directory containing STL. If directory, a random STL file will be loaded from the directory.
    zoom : bool, optional
        If True, plot will be zoomed in to show the particle as large as  Defaults to True

    Returns
    -------
    matplotlib.figure, matplotlib.axis
        Matplotlib figure and axis objects corresponding to 3D plot
    """
    stl_path = Path(stl_path)
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
        ax.set_xlim(np.min(stl_mesh.vectors.T[0]), np.max(stl_mesh.vectors.T[0]))
        ax.set_ylim(np.min(stl_mesh.vectors.T[1]), np.max(stl_mesh.vectors.T[1]))
        ax.set_zlim(np.min(stl_mesh.vectors.T[2]), np.max(stl_mesh.vectors.T[2]))
    return fig, ax

def plot_particle_slices(imgs_single_particle, n_slices=4, fig_w=7, dpi=100):
    """Plot a series of images of a single particle across n_slices number of slices.

    Parameters
    ----------
    imgs_single_particle : numpy.ndarray
        3D array of the same size as segment_dict['integer-labels'] that is only nonzero where pixels matched value of integer_label in original array
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
    print(f'Particle bounds: {bounds[0], bounds[3]}, {bounds[1], bounds[4]}, {bounds[2], bounds[5]}')

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
        ax[i].imshow(imgs_single_particle[slice_i, ...], interpolation='nearest')
        ax[i].set_axis_off()
        ax[i].set_title(f'Slice: {slice_i}')
    return fig, ax

def plot_imgs(imgs, n_imgs=3, fig_w=7.5, dpi=100):
    """Plot images.

    Parameters
    ----------
    imgs : list
        3D NumPy array or list of 2D arrays representing images to be plotted.
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
    n_rows = 1
    n_cols = n_imgs
    fig_h = fig_w * (img_h / img_w) * (n_rows / n_cols)
    fig, axes = plt.subplots(
        1, n_imgs, figsize=(fig_w, fig_h), constrained_layout=True, dpi=dpi, 
        facecolor='white'
    )
    if n_imgs == 1:
        axes.imshow(imgs, interpolation='nearest')
        axes.axis('off')
    else:
        ax = axes.ravel()
        spacing = total_imgs // n_imgs
        img_idcs = [i * spacing for i in range(n_imgs)]
        print(f'Plotting images: {img_idcs}')
        for i in range(n_imgs):
            idx = img_idcs[i]
            ax[i].imshow(imgs[idx, ...], interpolation='nearest')
            ax[i].axis('off')
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
        Dictionary containing segmentation routine steps, as returned from watershed_segment()
    img_idx : int
        Index of image on which particle labels will be shown
    label_color : str, optional
        Color of text of which labels will be shown, by default 'white'
    label_bg_color : tuple, optional
        Color of label background. Defaults to transparent RGBA tuple: (0, 0, 0, 0)
    use_color_labels : bool, optional
        If true, labels are converted to color labels to be plotted on image.
    fig_w : int, optional
        Width in inches of figure that will contain the labeled image, by default 7

    Returns
    -------
    matplotlib.figure, matplotlib.axis
        Matplotlib figure and axis objects corresponding to 3D plot
    """
    labels = segment_dict['integer-labels']
    regions = measure.regionprops(labels[img_idx, ...])
    label_centroid_pairs = [(region.label, region.centroid) for region in regions]
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
        3D NumPy array or list of 2D arrays representing preprocessed images to be plotted.
    imgs_binarized : list
        3D NumPy array or list of 2D arrays representing binarized images to be plotted.
    segment_dict : dict
        Dictionary containing segmentation routine steps, as returned from watershed_segment().
    n_imgs : int, optional
        Number of 2D images to plot from the 3D array. Defaults to 3.
    plot_maxima : int, optional
        If true, maxima used to seed watershed segmentation will be plotted on distance map. Defaults to True.
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
        img_idcs = [i * spacing for i in range(n_imgs)]
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
    
def ct_to_stl_files_workflow(
    ct_img_dir,
    stl_dir_location,
    spatial_res=1,
    min_peak_distance='median', 
    slice_lims=None,
    row_lims=None,
    col_lims=None,
    file_suffix='tiff'
):
    """Workflow function that takes loads CT images, segments them, converts each segmented particle to a triangular mesh, and saves that mesh as an STL file.

    Parameters
    ----------
    ct_img_dir : Path or str
        Path of directory containing CT images.
    stl_dir_location : Path or str
        Location where directory will be created to hold STL files.
    spatial_res : float, optional
        Factor to apply to multiply spatial vectors of saved STL. Applying the spatial/pixel resolution of the CT scan will give the STL file units of the value. Defaults to 1 to save the STL in units of pixels.
    min_peak_distance : str or int, optional
        Minimum distance between distance map maxima to be used in watershed segmentation. Can be 'median' to use two times the equivalent radius of the median particle. Defaults to 'median_radius'
    slice_lims : tuple, optional
        Range of image slices to be loaded, by default None
    row_lims : tuple, optional
        Range of rows in images to be loaded, by default None
    col_lims : tuple, optional
        Range of columns in images to be loaded, by default None
    file_suffix : str, optional
        File suffixes of images in ct_img_dir, by default 'tiff'
    """
    # Load images as 3D array from a directory containing images
    print('Loading images...')
    imgs, scan_name = load_images(
        ct_img_dir,
        slice_crop=slice_lims,
        row_crop=row_lims, 
        col_crop=col_lims,
        return_3d_array=True,
        convert_to_float=True,
        also_return_dir_name=True,
        file_suffix=file_suffix
    )
    print(f'Images loaded as 3D array: {imgs.shape}')
    print('Binarizing images...')
    # Binarize data
    imgs_binarized, thresh_vals = binarize_multiotsu(imgs, n_otsu_classes=2)
    print('Segmenting images...')
    # Segment particles
    segment_dict = watershed_segment(
        imgs_binarized, 
        min_peak_distance=min_peak_distance,
        return_dict=True
    )
    print('Saving segmented images as STL files...')
    # Save each segmented particle as a separate STL file
    save_as_stl_files(
        stl_dir_location, 
        segment_dict, 
        scan_name,
        spatial_res=spatial_res,
        return_dir_path=False
    )

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
        fatalError('Error in command-line arguments.  Enter ./segment.py -h for more help')
    yamlFile = ''
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        if opt == "-f":
            yamlFile = str(arg)

    #---------------------
    # Read YAML input file
    #---------------------
    if yamlFile == '':
        fatalError('No input file specified.  Try ./segment.py -h for more help.')
    stream = open(yamlFile, 'r')
    UI = yaml.load(stream,Loader=yaml.FullLoader)   # User Input
    stream.close()
    ui_ct_img_dir           = UI['Files']['CT Scan Dir']
    ui_stl_dir_location     = UI['Files']['STL Dir']
    ui_output_filename_base = UI['Files']['STL Prefix']
    ui_stl_overwrite        = UI['Files']['Overwrite Existing STL Files']
    ui_single_particle_iso  = UI['Files']['Particle ID']
    ui_suppress_save_msg    = UI['Files']['Suppress Save Messages']
    ui_file_suffix          = UI['Load']['File Suffix']
    ui_slice_crop           = UI['Load']['Slice Crop']
    ui_row_crop             = UI['Load']['Row Crop']             
    ui_col_crop             = UI['Load']['Col Crop']
    ui_use_median_filter    = UI['Preprocess']['Apply Median Filter']
    ui_rescale_range        = UI['Preprocess']['Rescale Intensity Range']
    ui_n_otsu_classes       = UI['Binarize']['Number of Otsu Classes']
    ui_n_selected_classes   = UI['Binarize']['Number of Classes to Select']
    ui_use_int_dist_map     = UI['Segment']['Use Integer Distance Map']
    ui_min_peak_distance    = UI['Segment']['Min Peak Distance']
    ui_exclude_borders      = UI['Segment']['Exclude Border Particles']
    ui_erode_particles      = UI['STL']['Erode Particles']    
    ui_voxel_step_size      = UI['STL']['Marching Cubes Voxel Step Size']    
    ui_spatial_res          = UI['STL']['Pixel-to-Length Ratio']
    ui_show_segment_fig     = UI['Plot']['Show Segmentation Figure']
    ui_n_imgs               = UI['Plot']['Number of Images']
    ui_plot_maxima          = UI['Plot']['Plot Maxima']
    ui_show_label_fig       = UI['Plot']['Show Particle Labels Figure']
    ui_label_idx            = UI['Plot']['Particle Label Image Index']
    ui_show_stl_fig         = UI['Plot']['Show Random STL Figure']

    #------------
    # Load images
    #------------
    print()
    print('Loading images...')
    imgs = load_images(
        ui_ct_img_dir,
        slice_crop=ui_slice_crop,
        row_crop=ui_row_crop,
        col_crop=ui_col_crop,
        convert_to_float=True,
        file_suffix=ui_file_suffix
    )
    print('--> Images loaded as 3D array: ', imgs.shape)
    print('--> Size of array (GB): ', imgs.nbytes / 1E9)

    #------------------
    # Preprocess images
    #------------------
    print()
    print('Preprocessing images...')
    imgs_pre = preprocess(
        imgs, median_filter=ui_use_median_filter, 
        rescale_intensity_range=ui_rescale_range
    )
    print('--> Preprocessing complete')
    print('--> Size of array (GB): ', imgs_pre.nbytes / 1E9)

    #----------------
    # Binarize images
    #----------------
    print()
    print('Binarizing images...')
    imgs_binarized, thresh_vals = binarize_multiotsu(
        imgs_pre, n_otsu_classes=ui_n_otsu_classes, 
        n_selected_thresholds=ui_n_selected_classes, 
    )
    print('--> Binarization complete')
    print('--> Size of array (GB): ', imgs_binarized.nbytes / 1E9)

    #---------------
    # Segment images
    #---------------
    print()
    print('Segmenting images...')
    segment_dict = watershed_segment(
        imgs_binarized, min_peak_distance=ui_min_peak_distance, 
        use_int_dist_map=ui_use_int_dist_map, return_dict=True
    )
    # Count number of particles segmented
    n_particles = np.max(segment_dict['integer-labels'])
    n_particles_digits = len(str(n_particles))
    print(f'--> Segmentation complete. {n_particles} particle(s) segmented.')
    # sys.getsizeof() doesn't represent nested objects; need to add manually
    print('--> Size of segmentation results (GB):')
    print(f'----> Dictionary: {sys.getsizeof(segment_dict) / 1E9}')
    for key, val in segment_dict.items():
        print(f'----> {key}: {sys.getsizeof(val) / 1E9}')

    if ui_exclude_borders:
        # How Many Particles Were Segmented?
        n_particles = np.max(segment_dict['integer-labels'])
        n_particles_digits = len(str(n_particles))
        print('--> Number of particles before border exclusion: ', str(n_particles))
        print()
        print('Excluding border particles...')
        segment_dict['integer-labels'] = segmentation.clear_border(
            segment_dict['integer-labels']
        )
    regions = measure.regionprops(segment_dict['integer-labels'])
    n_particles_noborder = len(regions)
    print('--> Number of particles: ', str(n_particles_noborder))
    
    #---------------------------------------
    # Create Surface Meshes of Each Particle 
    #---------------------------------------
    print()
    print('Generating surface meshes...')
    ui_stl_dir_location = Path(ui_stl_dir_location)
    if not ui_stl_dir_location.exists():
        print('--> Making directory: ', ui_stl_dir_location)
        ui_stl_dir_location.mkdir()
    for region in regions:
        # 3D area is actually volume (N voxels)
        n_voxels = region.area
        # Get bounding slice, row, and column
        min_slice, min_row, min_col, max_slice, max_row, max_col = region.bbox
        # Isolate Individual Particles
        imgs_particle = region.image
        if ui_erode_particles:
            imgs_particle = morphology.binary_erosion(imgs_particle)
        # Do Surface Meshing - Marching Cubes
        verts, faces, normals, values = measure.marching_cubes(
            imgs_particle, step_size=ui_voxel_step_size
        )
        # Create save path
        fn = (
            f'{ui_output_filename_base}'
            f'{str(region.label).zfill(n_particles_digits)}.stl'
        )
        stl_save_path = Path(ui_stl_dir_location) / fn
        # Calculate offsets for STL coordinates
        if ui_col_crop is not None:
            x_offset = ui_col_crop[0] + min_col
        else: 
            x_offset = min_col
        if ui_row_crop is not None:
            y_offset = ui_row_crop[0] + min_row
        else: 
            y_offset = min_row
        if ui_slice_crop is not None:
            z_offset = ui_slice_crop[0] + min_slice
        else:
            z_offset = min_slice
        # Save STL
        if ui_stl_overwrite and stl_save_path.exists():
            stl_save_path.unlink()
        save_stl(
            stl_save_path, verts, faces, spatial_res=ui_spatial_res, 
            suppress_save_message=ui_suppress_save_msg, 
            x_offset=x_offset, y_offset=y_offset, z_offset=z_offset
        )
    print('--> All .stl files written!')

    if ui_show_segment_fig:
        fig_seg_steps, axes_seg_steps = plot_segment_steps(
            imgs, imgs_pre, imgs_binarized, segment_dict, n_imgs=ui_n_imgs, 
            plot_maxima=ui_plot_maxima
        )
    if ui_show_label_fig:
        fig_labels, ax_labels = plot_particle_labels(
            segment_dict, ui_label_idx
        )
    if ui_show_stl_fig:
        fig_stl, ax_stl = plot_stl(ui_stl_dir_location)
    if ui_show_segment_fig or ui_show_label_fig or ui_show_stl_fig:
        plt.show()
    

if __name__ == '__main__':
    os.system('clear')
    print('')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to SegmentFlow!')
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
        
