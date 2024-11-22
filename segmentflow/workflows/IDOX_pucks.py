import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view, mesh
from segmentflow.workflows.workflow import Workflow
from skimage import color, draw, feature, filters, transform, util
import sys
import numpy as np


class IDOX_pucks(Workflow):
    def __init__(self, yaml_path=None, args=None):
        # Initialize parent class to set yaml_path
        super().__init__(yaml_path=yaml_path, args=args)
        self.name = Path(__file__).stem
        self.description = (
            'This workflow segments IDOX particles from a binder in CT scans'
            ' of pressed pucks taken with the CSM Zeiss Versa x-ray source.'
            ' Manual thresholds are set for semantic segmentation.'
            ' The output can be labeled voxels following semantic segmentation'
            ' instance segmentation, and/or STL files corresponding to each'
            ' instance-segmented particle.'
            ' Developed for Segmentflow v0.0.5.'
        )
        self.categorized_input_shorthands = {
            'A. Input' : {
                'in_dir_path'  : '01. Input dir path',
                'file_suffix'  : '02. File suffix',
                'slice_crop'   : '03. Slice crop',
                'row_crop'     : '04. Row crop',
                'col_crop'     : '05. Column crop',
                'spatial_res'  : '06. Pixel size',
            },
            'B. Output' : {
                'out_dir_path' :
                    '01. Path to save output dir',
                'overwrite' :
                    '02. Overwrite files',
                'out_prefix' :
                    '03. Output prefix',
                'slices' :
                    '04. Specify slices to plot',
                'nslices' :
                    '05. Number of slices in checkpoint plots',
                'save_stls' :
                    '06. Save STL files',
                'suppress_save_msg' :
                    '07. Suppress save message for each STL file',
                'save_voxels' :
                    '08. Save voxel TIF stack'
            },
            'C. Preprocessing' : {
                'pre_seg_rad_filter' :
                    '01. Apply radial filter',
                'pre_seg_med_filter' :
                    '02. Apply median filter',
                'rescale_range' :
                    '03. Range for rescaling intensity (percentile)',
            },
            'D. Segmentation' : {
                'thresh_vals' :
                    '01. Threshold values for semantic segmentation',
                'thresh_hist_ylims' :
                    '02. Upper and lower y-limits of histogram',
                'fill_holes' :
                    '03. Fill holes in semantic segmentation',
                'min_vol' :
                    '04. Min particle volume saved (voxels)',
                'perform_seg' :
                    '05. Perform instance segmentation',
                'min_peak_dist' :
                    '06. Min distance between region centers (pixels)',
                'exclude_borders' :
                    '07. Exclude border particles',
            },
            'E. Surface Meshing' : {
                'n_erosions' :
                    '01. Number of pre-surface meshing erosions',
                'post_seg_med_filter' :
                    '02. Smooth voxels with median filtering',
                'voxel_step_size' :
                    '03. Marching cubes voxel step size',
                'mesh_smooth_n_iters' :
                    '04. Number of smoothing iterations',
                'mesh_simplify_n_tris' :
                    '05. Target number of triangles/faces',
                'mesh_simplify_factor' :
                    '06. Simplification factor per iteration',
            },
        }
        self.default_values = {
            'in_dir_path'          : 'REQUIRED',
            'file_suffix'          : 'tiff',
            'slice_crop'           : None,
            'row_crop'             : None,
            'col_crop'             : None,
            'spatial_res'          : 1,
            'out_dir_path'         : 'REQUIRED',
            'out_prefix'           : '',
            'overwrite'            : False,
            'nslices'              : 4,
            'slices'               : None,
            'save_stls'            : True,
            'suppress_save_msg'    : True,
            'save_voxels'          : False,
            'pre_seg_rad_filter'   : False,
            'pre_seg_med_filter'   : False,
            'rescale_range'        : None,
            'thresh_vals'          : [30000, 62000],
            'thresh_hist_ylims'    : [0, 2e7],
            'fill_holes'           : False,
            'min_vol'              : None,
            'perform_seg'          : True,
            'min_peak_dist'        : 6,
            'exclude_borders'      : False,
            'n_erosions'           : 0,
            'post_seg_med_filter'  : False,
            'voxel_step_size'      : 1,
            'mesh_smooth_n_iters'  : None,
            'mesh_simplify_n_tris' : None,
            'mesh_simplify_factor' : None,
        }

    def run(self):
        show_checkpoints = False
        checkpoint_save_dir = self.ui['out_dir_path']

        #-------------#
        # Load images #
        #-------------#
        imgs = segment.load_images(
            self.ui['in_dir_path'],
            slice_crop=self.ui['slice_crop'],
            row_crop=self.ui['row_crop'],
            col_crop=self.ui['col_crop'],
            convert_to_float=True,
            file_suffix=self.ui['file_suffix'],
            logger=self.logger
        )
        # Generate raw imgs viz
        fig, axes = view.vol_slices(
                imgs,
                slices=self.ui['slices'],
                nslices=self.ui['nslices'],
                print_slices=False,
                fig_w=7.5,
                dpi=300
            )
        fig_n = 0
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='raw-imgs')

        #-------------------#
        # Preprocess images #
        #-------------------#
        if self.ui['pre_seg_rad_filter']:
            #---------------#
            # Radial filter #
            #---------------#
            self.logger.info('Normalizing radial intesnity...')
            imgs = segment.radial_filter(imgs)
            fig, axes = view.vol_slices(
                imgs,
                slices=self.ui['slices'],
                nslices=self.ui['nslices'],
                print_slices=False,
                fig_w=7.5,
                dpi=300
            )
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='radial-filtered')

        #---------------------#
        # Intensity Rescaling #
        #---------------------#
        imgs_med = segment.preprocess(
            imgs,
            median_filter=self.ui['pre_seg_med_filter'],
            logger=self.logger
        )
        fig, ax = view.histogram(
            imgs_med, mark_percentiles=self.ui['rescale_range'])
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='intensity-rescale-hist')
        # Preprocess images
        imgs_pre = segment.preprocess(
            imgs,
            median_filter=False,
            rescale_intensity_range=self.ui['rescale_range'],
            logger=self.logger
        )
        imgs_pre = util.img_as_uint(imgs_pre)
        # Generate preprocessed viz
        fig, axes = view.vol_slices(
                imgs_pre,
                slices=self.ui['slices'],
                nslices=self.ui['nslices'],
                print_slices=False,
                fig_w=7.5,
                dpi=300
            )
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='preprocessed-imgs')

        #-----------------------#
        # Semantic segmentation #
        #-----------------------#
        # Sort threshold values in descending order
        # thresholds = sorted(self.ui['thresh_vals'], reverse=True)
        thresholds = sorted(self.ui['thresh_vals'])
        # Calc semantic seg threshold values and generate histogram
        fig, ax = view.histogram(imgs_pre, mark_values=thresholds)
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='semantic-seg-hist')
        # Segment images with threshold values
        imgs_semantic = segment.isolate_classes(imgs_pre, thresholds)
        # Calc particle to binder ratio (voxels)
        particles_to_binder = segment.calc_voxel_stats(
            imgs_semantic, logger=self.logger)
        # Generate semantic label viz
        fig, axes = view.vol_slices(
                imgs_semantic,
                slices=self.ui['slices'],
                nslices=self.ui['nslices'],
                print_slices=False,
                fig_w=7.5,
                dpi=300
            )
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='semantic-seg-imgs')

        #------------------#
        # Fill small holes #
        #------------------#
        if self.ui['fill_holes'] is not None:
            imgs_semantic = segment.fill_holes(imgs_semantic)
            # Calc particle to binder ratio (voxels)
            particles_to_binder = segment.calc_voxel_stats(
                imgs_semantic, logger=self.logger)
            # Generate semantic label viz
            fig, axes = view.vol_slices(
                    imgs_semantic,
                    slices=self.ui['slices'],
                    nslices=self.ui['nslices'],
                    print_slices=False,
                    fig_w=7.5,
                    dpi=300
                )
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='semantic-seg-imgs-holes-filled')

        #------------------------#
        # Remove Small Particles #
        #------------------------#
        if self.ui['min_vol'] is not None:
            imgs_semantic = segment.remove_particles(imgs_semantic, self.ui['min_vol'])
            # Generate semantic label viz with small particles removed
            fig, axes = view.vol_slices(
                    imgs_semantic,
                    slices=self.ui['slices'],
                    nslices=self.ui['nslices'],
                    print_slices=False,
                    fig_w=7.5,
                    dpi=300
                )
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='semantic-seg-small-removed')

        #----------------#
        # Segment images #
        #----------------#
        if self.ui['perform_seg']:
            # Clear up memory
            imgs = None
            imgs_med = None
            imgs_pre = None
            imgs_instance = segment.watershed_segment(
                imgs_semantic==2,
                min_peak_distance=self.ui['min_peak_dist'],
                exclude_borders=self.ui['exclude_borders'],
                return_dict=False,
                logger=self.logger
            )
            # Merge semantic and instance segs to represent binder and particles
            imgs_labeled = segment.merge_segmentations(
                imgs_semantic, imgs_instance)
            # Post-seg median filter
            if self.ui['post_seg_med_filter']:
                imgs_labeled = filters.median(imgs_labeled)
                imgs_instance[imgs_labeled == 1] = 0
            # Generate instance label viz
            fig, axes = view.color_labels(
                imgs_instance,
                slices=self.ui['slices'],
                nslices=self.ui['nslices'],
                fig_w=7.5,
                dpi=300
            )
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='instance-seg-imgs')

        #-------------#
        # Save voxels #
        #-------------#
        if self.ui['save_voxels']:
            if self.ui['perform_seg']:
                save_path = Path(self.ui['out_dir_path']) / (
                    f"{self.ui['out_prefix']}_labeled_voxels"
                )
                segment.save_images(
                    imgs_labeled,
                    save_path,
                    logger=self.logger
                )
            else:
                save_path = Path(self.ui['out_dir_path']) / (
                    f"{self.ui['out_prefix']}_semantic_voxels")
                segment.save_images(
                    imgs_semantic, save_path, logger=self.logger
                )

        #----------------------------------------#
        # Create Surface Meshes of Each Particle #
        #----------------------------------------#
        if self.ui['perform_seg'] and self.ui['save_stls']:
            # Clear up memory
            imgs_semantic = None
            imgs_instance = None
            # Remove voxels corresponding to binder to save only particles as STLs
            imgs_labeled[imgs_labeled == 1] = 0
            segment.save_as_stl_files(
                imgs_labeled,
                self.ui['out_dir_path'],
                self.ui['out_prefix'],
                suppress_save_msg=self.ui['suppress_save_msg'],
                slice_crop=self.ui['slice_crop'],
                row_crop=self.ui['row_crop'],
                col_crop=self.ui['col_crop'],
                stl_overwrite=self.ui['overwrite'],
                spatial_res=self.ui['spatial_res'],
                n_erosions=self.ui['n_erosions'],
                median_filter_voxels=self.ui['post_seg_med_filter'],
                voxel_step_size=self.ui['voxel_step_size'],
                logger=self.logger
            )
            # Generate figure showing fraction of STLs that are watertight
            fig, ax = view.watertight_fraction(
                Path(self.ui['out_dir_path'])
                / f"{self.ui['out_prefix']}_STLs/{self.ui['out_prefix']}_properties.csv"
            )
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='watertight-fraction')
            # Generate figure showing fraction of STLs that are watertight
            fig, ax = view.watertight_volume(
                Path(self.ui['out_dir_path'])
                / f"{self.ui['out_prefix']}_STLs/{self.ui['out_prefix']}_properties.csv"
            )
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='watertight-volume')

            #----------------------------------------------#
            # Postprocess surface meshes for each particle #
            #----------------------------------------------#
            if (
                self.ui['mesh_smooth_n_iters'] is not None
                or self.ui['mesh_simplify_n_tris'] is not None
                or self.ui['mesh_simplify_factor'] is not None
            ):
                # Iterate through each STL file, load the mesh, and smooth/simplify
                mesh.postprocess_meshes(
                    Path(self.ui['out_dir_path']) / f"{self.ui['out_prefix']}_STLs",
                    smooth_iter=self.ui['mesh_smooth_n_iters'],
                    simplify_n_tris=self.ui['mesh_simplify_n_tris'],
                    iterative_simplify_factor=self.ui['mesh_simplify_factor'],
                    recursive_simplify=False,
                    resave_mesh=False,
                    save_dir_path=(
                        Path(self.ui['out_dir_path'])
                        / f"{self.ui['out_prefix']}_processed-STLs",
                    )
                )


if __name__ == '__main__':
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    # Pass path to YAML from args and store in workflow class
    workflow = IDOX_pucks(args=sys.argv[1:])
    # Load input data from YAML and store as UI attribute
    workflow.read_yaml()
    # Create and store a logger object as an attribute for saving a log file
    workflow.create_logger()
    try:
        workflow.run()
    except Exception as error:
        workflow.logger.exception(error)
    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Successful Completion')
    print('~~~~~~~~~~~~~~~~~~~~~')
    print()

