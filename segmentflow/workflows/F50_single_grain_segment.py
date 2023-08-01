import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view, mesh
import sys


WORKFLOW_NAME = Path(__file__).stem

WORKFLOW_DESCRIPTION = (
    'This workflow segments F50 sand grains from a CT scan of a poured sample'
    ' and outputs a labeled TIF stack and/or STL files corresponding'
    ' to each segmented particle. Developed for v0.0.1.'
)

CATEGORIZED_INPUT_SHORTHANDS = {
    'Files' : {
        'in_dir_path'  : 'Input dir path',
        'file_suffix'  : 'File suffix',
        'slice_crop'   : 'Slice crop',
        'row_crop'     : 'Row crop',
        'col_crop'     : 'Column crop',
        'spatial_res'  : 'Pixel size',
        'out_dir_path' : 'Path to save output dir',
        'out_prefix'   : 'Output prefix',
        'overwrite'    : 'Overwrite files'
    },
    'View' : {
        'view_slices'   : 'Slices to view',
        'view_raw'      : 'View raw images',
        'view_pre'      : 'View preprocessed images',
        'view_semantic' : 'View semantic images',
        'view_labeled'  : 'View labeled images',
    },
    'Preprocess' : {
        'pre_seg_med_filter' : 'Apply median filter',
        'rescale_range'      : 'Rescale intensity range',
    },
    'Segmentation' : {
        'thresh_nbins'      : 'Histogram bins for calculating thresholds',
        'view_thresh_hist'  : 'View histogram with threshold values',
        'thresh_hist_ylims' : 'Upper and lower y-limits of histogram',
        'perform_seg'       : 'Perform instance segmentation',
        'min_peak_dist'     : 'Min peak distance',
        'exclude_borders'   : 'Exclude border particles',
        'save_voxels'       : 'Save labeled voxels'
    },
    'STL' : {
        'create_stls'          : 'Create STL files',
        'suppress_save_msg'    : 'Suppress save message for each STL file',
        'n_erosions'           : 'Number of pre-surface meshing erosions',
        'post_seg_med_filter'  : 'Smooth voxels with median filtering',
        'voxel_step_size'      : 'Marching cubes voxel step size',
        'mesh_smooth_n_iters'  : 'Number of smoothing iterations',
        'mesh_simplify_n_tris' : 'Target number of triangles/faces',
        'mesh_simplify_factor' : 'Simplification factor per iteration',
    },
}

DEFAULT_VALUES = {
    'in_dir_path'          : 'REQUIRED',
    'file_suffix'          : 'tiff',
    'slice_crop'           : None,
    'row_crop'             : None,
    'col_crop'             : None,
    'out_dir_path'         : 'REQUIRED',
    'out_prefix'           : '',
    'overwrite'            : False,
    'view_slices'          : True,
    'view_raw'             : True,
    'view_pre'             : True,
    'view_semantic'        : True,
    'view_labeled'         : True,
    'pre_seg_med_filter'   : False,
    'rescale_range'        : None,
    'thresh_nbins'         : 256,
    'view_thresh_hist'     : True,
    'thresh_hist_ylims'    : [0, 2e7],
    'perform_seg'          : True,
    'min_peak_dist'        : 6,
    'exclude_borders'      : True,
    'save_voxels'          : True,
    'create_stls'          : True,
    'suppress_save_msg'    : True,
    'n_erosions'           : 0,
    'post_seg_med_filter'  : False,
    'spatial_res'          : 1,
    'voxel_step_size'      : 1,
    'mesh_smooth_n_iters'  : None,
    'mesh_simplify_n_tris' : None,
    'mesh_simplify_factor' : None,
    'seg_fig_show'         : False,
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
        imgs, median_filter=ui['pre_seg_med_filter'],
        rescale_intensity_range=ui['rescale_range']
    )

    #-----------------------#
    # Semantic segmentation #
    #-----------------------#
    print()
    if ui['view_thresh_hist']:
        thresholds, thresh_fig, thresh_ax = segment.threshold_multi_min(
            imgs_pre, nbins=ui['thresh_nbins'], return_fig_ax=True,
            ylims=ui['thresh_hist_ylims']
        )
    else:
        thresholds = segment.threshold_multi_min(
            imgs_pre, nbins=ui['thresh_nbins'], return_fig_ax=False,
        )
    imgs_semantic = segment.isolate_classes(imgs_pre, thresholds)
    if ui['view_semantic']:
        fig, axes = view.plot_slices(
                imgs_semantic,
                slices=ui['view_slices'],
                print_slices=False,
                fig_w=7.5,
                dpi=100
            )

    #-----------------------#
    # Instance segmentation #
    #-----------------------#
    if ui['perform_seg']:
        print()
        imgs = None
        imgs_pre = None
        imgs_labeled = segment.watershed_segment(
            imgs_semantic==len(thresholds),
            min_peak_distance=ui['min_peak_dist'],
            exclude_borders=ui['exclude_borders'],
            return_dict=False
        )
        # Merge semantic and instance segmentations
        imgs_labeled = segment.merge_segmentations(imgs_semantic, imgs_labeled)
        if ui['view_labeled']:
            fig, axes = view.plot_color_labels(
                imgs_labeled,
                slices=ui['view_slices'],
                fig_w=7.5,
                dpi=100
            )
        if ui['save_voxels']:
            segment.save_images(
                imgs_labeled,
                Path(ui['out_dir_path']) / f"{ui['out_prefix']}_labeled_voxels"
            )

    #----------------------------------------#
    # Create Surface Meshes of Each Particle #
    #----------------------------------------#
    if ui['perform_seg'] and ui['create_stls']:
        print()
        segment.save_as_stl_files(
            imgs_labeled,
            ui['out_dir_path'],
            ui['out_prefix'],
            suppress_save_msg=ui['suppress_save_msg'],
            slice_crop=ui['slice_crop'],
            row_crop=ui['row_crop'],
            col_crop=ui['col_crop'],
            stl_overwrite=ui['overwrite'],
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
    if (
        ui['view_raw'] or ui['view_pre'] or ui['view_semantic']
        or ui['view_labeled']
    ):
        plt.show()

# Converted script
def converted_script():
    # File paths
    ct_dir_path = Path(r'/Users/erikjensen/Documents/PSAAP/Working/UQ_singleParticleCompression/F50_1/Scan_1/')
    save_dir_path = Path('/Users/erikjensen/Documents/PSAAP/Working/UQ_singleParticleCompression/F50_1/stls/')
    save_tiff_dir_path = Path('/Users/erikjensen/Documents/PSAAP/Working/UQ_singleParticleCompression/F50_1/tiffs/')
    save_images = False

    ct_dir_path.exists()
    test = [path for path in ct_dir_path.glob('*.tif')]
    len(test)

    #-------------#
    # Load images #
    #-------------#
    imgs = segment.load_images(
        ct_dir_path,
        slice_crop=[0, 266],
        row_crop=[600, 1250],
        col_crop=[600, 1250],
        convert_to_float=True,
        file_suffix='.tif'
    )
    # row & col crop deterined in NB 14
    slices = [0, 100, 200, 265]
    fig, axes = view.plot_slices(
        imgs,
        slices=slices,
        fig_w=7.5,
        dpi=100
    )

    #---------------#
    # Median filter #
    #---------------#
    imgs_med = segment.preprocess(
        imgs, median_filter=True,
        rescale_intensity_range=None
    )

    imgs_binarized, thresh_vals = segment.binarize_multiotsu(
        imgs_med, n_otsu_classes=2
    )
    # Plot histogram
    hist, hist_centers = exposure.histogram(imgs_med)
    fig, ax = plt.subplots(dpi=150)
    ax.plot(hist_centers, hist, lw=1)
    for val in thresh_vals:
        ax.axvline(val, c='red', zorder=0)
    fig, axes = view.plot_slices(
        imgs_binarized,
        slices=slices,
        fig_w=7.5,
        dpi=100
    )

    # ## Crop grain between artifact-heavy slices and remove noise

    # zyx
    print(f'{imgs_binarized.shape=}')
    # yxz
    imgs_binarized_zx = np.rot90(imgs_binarized, axes=(2, 0))
    print(f'{imgs_binarized_zx.shape=}')

    fig = px.imshow(
        imgs_binarized_zx, binary_string=True, animation_frame=0
    )
    fig.show()

    #imgs_cropped = imgs_binarized[15:245, ...]
    #imgs_cropped = imgs_binarized[0:257, ...] #- v2
    imgs_cropped = imgs_binarized[9:260, ...] #- v3
    imgs_cleaned = np.zeros_like(imgs_cropped)
    for n in range(imgs_cropped.shape[0]):
        imgs_cleaned[n, ...] = morphology.remove_small_objects(
            measure.label(imgs_cropped[n, ...]), min_size=500).astype(bool)
    fig, axes = view.plot_slices(
        imgs_cleaned,
        slices=np.linspace(0, len(imgs_cropped) - 1, 8).astype(int),
        imgs_per_row=4,
        fig_w=7.5,
        dpi=100
    )

    imgs_filled = np.zeros_like(imgs_cleaned)
    for n in range(imgs_cleaned.shape[0]):
        imgs_filled[n, ...] = ndi.binary_fill_holes(imgs_cleaned[n, ...])
    fig, axes = view.plot_slices(
        imgs_filled,
        slices=np.linspace(0, len(imgs_cropped) - 1, 8).astype(int),
        imgs_per_row=4,
        fig_w=7.5,
        dpi=100
    )

    #Added to help remove the artifacts - Erode and then "re"-rode
    imgs_eroded = ndi.binary_erosion(
        imgs_filled,
        iterations=8
    )
    imgs_eroded = ndi.binary_dilation(
        imgs_eroded,
        iterations=8
    )
    fig, axes = view.plot_slices(
        imgs_eroded,
        slices=np.linspace(0, len(imgs_cropped) - 1, 8).astype(int),
        imgs_per_row=4,
        fig_w=7.5,
        dpi=100
    )

    fig = px.imshow(
        imgs_eroded,
        binary_string=True, animation_frame=0
    )
    fig.show()

    fig = px.imshow(
        np.rot90(imgs_eroded, axes=(2, 0)),
        binary_string=True, animation_frame=0
    )
    fig.show()


    # ## Convert voxels to STL

    # If max value of labeled images is not 1, there is more than one connected
    # particle and the largest needs to be isolated from the rest

    #imgs_filled_labeled = measure.label(imgs_filled) #Removed when added in the erode/re-rode bits
    imgs_filled_labeled = measure.label(imgs_eroded)
    print('Number of particles =', imgs_filled_labeled.max())
    if imgs_filled_labeled.max() > 1:
        print('Isolating the largest particle...')
        df = pd.DataFrame(measure.regionprops_table(
            imgs_filled_labeled, properties=['label', 'area', 'bbox']
        ))
        df = df.rename(columns={'area' : 'volume'})
        # Get the label according to the particle with the largest volume
        largest_label = df.loc[df.volume.idxmax(), 'label']
        imgs_largest_only = np.zeros_like(imgs_filled_labeled, dtype=np.ubyte)
        imgs_largest_only[imgs_filled_labeled == largest_label] = 1
        imgs_filled_labeled = imgs_largest_only
    # Save largest particle as STL - Resolution = 1.09 micrometers per pixel (0.00109 mm per pixel)
    segment.save_as_stl_files(
        imgs_filled_labeled,
        save_dir_path,
        f'{ct_dir_path.stem}-',
        make_new_save_dir=False,
        spatial_res=0.00109,
        stl_overwrite=True
    )

    segment.save_images(
        imgs_filled_labeled,
        save_tiff_dir_path
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

