import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import numpy as np
from pathlib import Path
from skimage import color, exposure, measure
from stl import mesh


#~~~~~~~~~~~~~~~~~~~~#
# Plotting Functions #
#~~~~~~~~~~~~~~~~~~~~#

def get_colors(n_colors, min=0, max=1, cmap=mpl.cm.gist_rainbow):
    """Helper function to generate a list of colors from a matplotlib colormap.
    ----------
    Parameters
    ----------
    val : int or float
        Number between vmin and vmax representing the position of the returned
        color in the colormap.
    min : int or float, optional
        Lower bound of range to which the colormap will be normalized.
        Defaults to 0.
    max : int or float, optional
        Upper bound of range to which colormap will be normalized.
        Defaults to 10.
    cmap : matplotlib colormap object, optional
        Colormap from which a color will be taken.
        Defaults to mpl.cm.gist_rainbow
    -------
    Returns
    -------
    list
        List of 4-tuples representing RGBA floats from cmap.
    """
    colors = []
    for i in np.linspace(min, max, n_colors):
        norm = mpl.colors.Normalize(vmin=min, vmax=max)
        color = cmap(norm(i))
        colors.append(color)
    return colors

def plot_hist(imgs, view_slice_i, hist_extent='stack', figsize=(8, 3), dpi=150):
    """Calculate and plot histogram for image(s).
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D image array representing stack of 2D MxN images.
    view_slice_i : int
        Index of slice that will be plotted beside histogram
    hist_extent : str, optional
        Extent of calculated histogram. Must be one of the following:
        - 'stack' : Calculate histogram of all slices (default)
        - 'slice' : Calculate histogram of view slice only
    figsize : 2-tuple, optional
        Size of figure in inches. Defaults to (8, 3).
    dpi : int
        Resolution of figure in dpi. Defaults to 150.
    ------
    Raises
    ------
    ValueError
        Raised if hist_extent is not 'stack' or 'slice'
    -------
    Returns
    -------
    matplotlib.Figure, matploblib.Axes
        Matplotlib figure and axes objects of generated figure.
    """
    # Calculate histogram
    img = imgs[view_slice_i, ...]
    if hist_extent == 'stack':
        hist, hist_centers = exposure.histogram(imgs)
    elif hist_extent == 'slice':
        hist, hist_centers = exposure.histogram(img)
    else:
        raise ValueError(
            "hist_type must be either 'stack' or 'slice'"
        )
    # Plot histogram
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, constrained_layout=True
    )
    ax = axes.ravel()
    ax[0].imshow(img, interpolation='nearest')
    ax[0].axis('off')
    ax[1].plot(hist_centers, hist, lw=1)
    return fig, ax

def plot_images(
    imgs,
    imgs_per_row=None,
    fig_w=7.5,
    dpi=100
):
    """Plot images.
    ----------
    Parameters
    ----------
    imgs : list
        List of NumPy arrays representing images to be plotted.
    imgs_per_row : int or None, optional
        Number of images to plot in each row. Default is None and all images
        are plotted in the same row.
    fig_w : float, optional
        Width of figure in inches, by default 7.5
    dpi : float, optional
        Resolution (dots per inch) of figure. Defaults to 300.
    -------
    Returns
    -------
    matplotlib.Figure, matplotlib.Axis
        2-tuple containing matplotlib figure and axes objects
    """
    if not isinstance(imgs, list):
        raise ValueError('Images must be passed as a list.')
    n_imgs = len(imgs)
    img_w = imgs[0].shape[1]
    img_h = imgs[0].shape[0]
    if imgs_per_row is None:
        n_cols = n_imgs
    else:
        n_cols = imgs_per_row
    n_rows = int(math.ceil( n_imgs / n_cols ))
    fig_h = fig_w * (img_h / img_w) * (n_rows / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True,
        dpi=dpi, facecolor='white'
    )
    if isinstance(axes, np.ndarray):
        ax = axes.ravel()
    else:
        # When only one image, wrap axis object into list to make iterable
        ax = [axes]
    for i, img in enumerate(imgs):
        ax[i].imshow(img, interpolation='nearest')
    for a in ax:
        a.axis('off')
    return fig, axes

def plot_mesh_3D(verts, faces):
    """Plot triangualar mesh with Matplotlib.
    ----------
    Parameters
    ----------
    verts : array-like
        Array of (x, y, z) vertices indexed with faces to construct triangles.
    faces : array-like
        Array of indices referencing verts that define the triangular faces of
        the mesh.
    -------
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
    ----------
    Parameters
    ----------
    stl_path : Path or str
        Path to an STL file to load or a directory containing STL. If
        directory, a random STL file will be loaded from the directory.
    zoom : bool, optional
        If True, plot will be zoomed in to show the particle as large as
        Defaults to True
    -------
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
            f'First parameter must string, pathlib.Path,'
            ' or stl.mesh.Mesh object. Object type: {type(path_or_mesh)}'
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
    ----------
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
    -------
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
    ----------
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
    -------
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
        n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True,
        dpi=dpi, facecolor='white'
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

def plot_sequences(image_sequences, fig_w=7.5, dpi=300, sequence_titles=None):
    """Plot image sequences in an MxN plot with M sequences of N frames each.
    Originally from Gus's Al-Si project.
    ----------
    Parameters
    ----------
    frame_sequences : list or dict
        List or dict of frame sequences to plot. If Dict, key will be plotted
        along the y-axis of the first image for each sequence.
    sequence_titles : list or str, optional
        List of strings to use as titles for sequences in plot. If 'keys',
        keys from dictionary will be used.
    fig_w : float
        Width in inches for figure. Defaults to 7.5
    dpi : float, optional
        Resolution (dots per inch) of figure. Defaults to 300.
    -------
    Returns
    -------
    fig : matplotlib.Figure
        Matplotlib figure object containing Axes with plots
    axes : numpy.ndarray
        Array of matplotlib Axes objects for each axis
    """
    # Set sequence_titles to dict keys if sequence_titles is not the string
    # 'none'
    if isinstance(image_sequences, dict):
        if sequence_titles == 'keys':
            sequence_titles = list(image_sequences.keys())
        image_sequences = list(image_sequences.values())
    n_axes_h = len(image_sequences)
    n_axes_w = len(image_sequences[0])
    img_w = image_sequences[0][0].shape[1]
    img_h = image_sequences[0][0].shape[0]
    fig_h = fig_w * (img_h / img_w) * (n_axes_h / n_axes_w)
    if sequence_titles is not None:
        # Trim a bit off the figure for each row to account for additional
        # width from titles
        fig_h -= 0.02 * n_axes_h
    fig, axes = plt.subplots(
        n_axes_h, n_axes_w, figsize=(fig_w, fig_h), constrained_layout=True,
        sharey=True, dpi=dpi, facecolor='white'
    )
    for i, frames in enumerate(image_sequences):
        for j, frame in enumerate(frames):
            axes[i, j].imshow(frame, vmin=0, vmax=1, interpolation='nearest')
            # axes[i, j].set_axis_off()
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticks([])
            axes[i, j].set_xticks([])
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['bottom'].set_visible(False)
            axes[i, j].spines['left'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)
        if sequence_titles is not None:
            axes[i, 0].set_ylabel(sequence_titles[i])
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
    """Plot segmented particles.
    ----------
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
    -------
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
    ----------
    Parameters
    ----------
    imgs : list
        3D NumPy array or list of 2D arrays representing images to be plotted.
    imgs_pre : list
        3D NumPy array or list of 2D arrays representing preprocessed images to
        be plotted.
    imgs_binarized : list
        3D NumPy array or list of 2D arrays representing binarized images to
        be plotted.
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
    -------
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

def plot_slices(
    imgs,
    nslices=3,
    slices=None,
    print_slices=True,
    imgs_per_row=None,
    fig_w=7.5,
    dpi=100
):
    """Plot slices of a 3D array representing a 3D volume.
    ----------
    Parameters
    ----------
    imgs : list
        3D NumPy array or list of 2D arrays representing images to be plotted.
    fig_w : float, optional
        Width of figure in inches, by default 7.5
    nslices : int, optional
        Number of slices to plot from 3D array. Defaults to 3.
    slices : None or list, optional
        Slice numbers to plot. Replaces n_imgs. Defaults to None.
    print_slices : bool, optional
        If True, print the slices being plotted. Defaults to True.
    dpi : float, optional
        Resolution (dots per inch) of figure. Defaults to 300.
    -------
    Returns
    -------
    matplotlib.Figure, matplotlib.Axis
        2-tuple containing matplotlib figure and axes objects
    """
    vmin = imgs.min()
    vmax = imgs.max()
    dim = len(imgs.shape)
    if dim == 2:
        nslices = 1
        total_imgs = 1
        img_w = imgs.shape[1]
        img_h = imgs.shape[0]
    else:
        total_imgs = imgs.shape[0]
        img_w = imgs[0].shape[1]
        img_h = imgs[0].shape[0]
    if slices is None:
        spacing = total_imgs // nslices
        img_idcs = [i * spacing for i in range(nslices)]
    else:
        nslices = len(slices)
        img_idcs = slices
    if imgs_per_row is None:
        n_cols = nslices
    else:
        n_cols = imgs_per_row
    n_rows = int(math.ceil( nslices / n_cols ))
    fig_h = fig_w * (img_h / img_w) * (n_rows / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True,
        dpi=dpi, facecolor='white'
    )
    if nslices == 1:
        axes.imshow(imgs, interpolation='nearest')
        axes.axis('off')
    else:
        ax = axes.ravel()
        if print_slices:
            print(f'Plotting images: {img_idcs}')
        for i, idx in enumerate(img_idcs):
            ax[i].imshow(
                imgs[idx, ...], vmin=vmin, vmax=vmax, interpolation='nearest')
        # Separated from loop in the that axes are left blank (un-full row)
        for a in ax:
            a.axis('off')
    return fig, axes

