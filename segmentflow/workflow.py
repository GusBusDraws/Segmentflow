# Local imports
import segment
import view
import mesh


#~~~~~~~~~~#
# Workflow #
#~~~~~~~~~~#

def segmentation_workflow(argv):

    #----------------------------#
    # Get command-line arguments #
    #----------------------------#
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

    #----------------------#
    # Read YAML input file #
    #----------------------#
    if yaml_file == '':
        fatalError(
            'No input file specified. Try ./segment.py -h for more help.'
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
        save_isolated_classes(imgs_pre, thresh_vals, ui['stl_dir_location'])

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

