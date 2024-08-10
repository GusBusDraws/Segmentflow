import imageio.v3 as iio
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from segmentflow import segment, view
from segmentflow.workflows.workflow import Workflow
from scipy import ndimage, spatial
from skimage import measure, morphology, segmentation
import sys


class SEM_outlines(Workflow):
    def __init__(self, yaml_path=None, args=None):
        # Initialize parent class to set yaml_path
        super().__init__(yaml_path=yaml_path, args=args)
        self.name = Path(__file__).stem
        self.description = (
            'This workflow takes a 2D Back-Scattered Electron (BSE) image from'
            ' a Scanning Electron Microscrope (SEM), segments the classes'
            ' (grain, binder, or void) according to provided values, instance'
            ' segments the grains with a watershed algorithm, selects the'
            ' bounding coordinates of each segmented grain, ensures the'
            ' bounding coordinates can create a loop, and orders the loop'
            ' according to increasing polar angle, and saves the resulting'
            ' ordered coordinates in CSV files (one per grain) and a separate'
            ' CSV file containing the bounding boxes of each grain.'
            ' Developed for Segmentflow v0.0.4.'
        )
        self.categorized_input_shorthands = {
            'A. Input' : {
                'sem_path'          : '01. Path to SEM image',
                'row_crop'          : '02. Row Crop',
                'col_crop'          : '03. Column Crop',
                'spatial_res'       : '04. Pixel size (ums)',
                'thresh_vals'       : '05. Threshold values',
                'merge_aid'         : '06. Open merging aid figure',
                'merge_path'        : '07. Path to TXT file defining merges',
                'exclude_borders'   : '08. Exclude regions along border',
                'smooth'            : '09. Smooth regions',
            },
            'B. Output' : {
                'out_dir_path'  : '01. Output dir path',
                'out_prefix'    : '02. Output prefix',
                'overwrite'     : '03. Overwrite files'
            },
        }
        self.default_values = {
            'sem_path'         : 'REQUIRED',
            'row_crop'         : 'REQUIRED',
            'col_crop'         : 'REQUIRED',
            'spatial_res'      : 1,
            'thresh_vals'      : None,
            'merge_aid'        : True,
            'merge_path'       : 'REQUIRED',
            'exclude_borders'  : False,
            'smooth'           : False,
            'out_dir_path'     : 'REQUIRED',
            'out_prefix'       : '',
            'overwrite'        : False,
        }

    def run(self):
        """Carry out workflow WORKFLOW_NAME as described by
        WORKFLOW_DESCRIPTION.
        """
        self.logger.info(f'Beginning workflow: {workflow.name}')
        show_checkpoints = False
        checkpoint_save_dir = self.ui['out_dir_path']

        #------------#
        # Load image #
        #------------#
        img_dir_path = self.ui['sem_path']
        img = iio.imread(img_dir_path)
        self.logger.info(f'Image loaded: {img.shape}')
        # Figure: SEM vs Crop
        fig, axes = plt.subplots(
            1, 2, dpi=300, facecolor='white', figsize=(8.6, 4),
            constrained_layout=True)
        axes[0].imshow(img, vmin=img.min(), vmax=img.max(), cmap='gray')
        row_crop = self.ui['row_crop']
        col_crop = self.ui['col_crop']
        rect = mpl.patches.Rectangle(
            (col_crop[0], row_crop[0]),
            col_crop[1]-col_crop[0], row_crop[1]-row_crop[0],
            linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title('Full SEM Image')
        img_crop = img[row_crop[0]:row_crop[1], col_crop[0]:col_crop[1]]
        axes[1].imshow(img_crop, vmin=img.min(), vmax=img.max(), cmap='gray')
        axes[1].set_title('Subarea')
        for a in axes:
            a.set_axis_off()
        fig_n = 0
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='subarea-crop')

        #-----------------------#
        # Semantic segmentation #
        #-----------------------#
        # Segment the classes (grain, binder, or void) according to input
        img_semantic = segment.isolate_classes(
            img_crop, self.ui['thresh_vals'], intensity_step=1)
        n_pixels = img_semantic.shape[0] * img_semantic.shape[1]
        n_void = np.count_nonzero(img_semantic == 0)
        n_binder = np.count_nonzero(img_semantic == 1)
        n_crystals = np.count_nonzero(img_semantic == 2)
        self.logger.info('Calculating pixel statistics...')
        self.logger.info(
            f'--> Void area fraction: {round(n_void / n_pixels, 3)}')
        self.logger.info(
            f'--> Crystal area fraction: {round(n_crystals / n_pixels, 3)}')
        self.logger.info(
            f'--> Crystal area fraction (void corrected):'
            f'{round(n_crystals / (n_pixels - n_void), 3)}')
        # Figure: Semantic segmentation
        fig, axes = view.histogram_and_semantic(
            img, self.ui['thresh_vals'], img_semantic)
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='semantic-seg')

        #-----------------------#
        # Instance segmentation #
        #-----------------------#
        # Segment the grains with a watershed algorithm
        img_labeled = segment.watershed_segment(
            img_semantic==2, min_peak_distance=2, logger=self.logger)
        img_colors = view.color_labels(img_labeled, return_image=True)
        if self.ui['exclude_borders']:
            self.logger.info('Clearing regions along image border...')
            # Clear segmentation border
            img_labeled = segmentation.clear_border(img_labeled)
        # Figure: Initial instance segmentation
        fig, axes = view.images(
            [img_crop, img_semantic, img_colors],
            fig_w=7.5, dpi=300)
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='instance-seg')

        #------------------#
        # Region merge aid #
        #------------------#
        # Open a figure that helps select regions to merge
        if self.ui['merge_aid']:
            fig, axes = view.images(
                [img_crop, img_labeled, img_colors],
                imgs_per_row=3, dpi=300)
            plt.show()
        else:
            #---------------#
            # Merge regions #
            #---------------#
            # Merge regions grouped by line in TXT file
            merge_labeled = segment.manual_merge(
                img_labeled, self.ui['merge_path'], logger=self.logger)
            merge_colors = view.color_labels(merge_labeled, return_image=True)
            # Figure: Brief figure showing merged regions
            imgs = [
                img_crop,
                img_colors,
                segmentation.mark_boundaries(
                    merge_colors, merge_labeled, mode='subpixel', color=(1,1,1))
            ]
            fig, axes = view.images(
                imgs, imgs_per_row=3, dpi=300, subplot_letters=True, fig_w=8)
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='merged-regions-detailed')
            # Figure: Detailed figure showing merged regions
            imgs = [
                img_crop,
                img_labeled,
                img_colors,
                segmentation.mark_boundaries(
                    merge_colors, merge_labeled, mode='subpixel', color=(1,1,1))
            ]
            fig, axes = view.images(
                imgs, imgs_per_row=4, dpi=300, subplot_letters=True, fig_w=8
            )
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='merged-regions-detailed')

            #-----------------------------#
            # Select bounding coordinates #
            #-----------------------------#
            viz_output = segment.save_bounding_coords(
                merge_labeled,
                self.ui['out_dir_path'],
                self.ui['out_prefix'],
                smooth=self.ui['smooth'],
                spatial_res=self.ui['spatial_res'],
                logger=self.logger,
                return_boundary_viz=True,
                return_smoothed_viz=self.ui['smooth'],
            )
            if self.ui['smooth']:
                subpixel_viz, smooth_viz = viz_output
            else:
                subpixel_viz = viz_output
            # Figure: Initial instance segmentation
            fig, axes = view.images([
                img_colors,
                view.color_labels(merge_labeled, return_image=True),
                view.color_labels(subpixel_viz, return_image=True),
            ], imgs_per_row=3, dpi=300, subplot_letters=True)
            fig_n += 1
            segment.output_checkpoints(
                fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                fn_n=fig_n, fn_suffix='region-bounds')
            if self.ui['smooth']:
                # Figure: Boundaries vs smoothed boundaries
                fig, axes = view.images([
                    view.color_labels(subpixel_viz, return_image=True),
                    view.color_labels(smooth_viz, return_image=True),
                ], imgs_per_row=2, dpi=300, subplot_letters=True)
                fig_n += 1
                segment.output_checkpoints(
                    fig, show=show_checkpoints, save_path=checkpoint_save_dir,
                    fn_n=fig_n, fn_suffix='bounds-smoothed')

            # Save a single CSV file containing the bounding boxes of all grains
            segment.save_bounding_boxes(
                merge_labeled, self.ui['out_dir_path'], self.ui['out_prefix'],
                self.ui['spatial_res']
            )

if __name__ == '__main__':
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    # Pass path to YAML from args and store in workflow class
    workflow = SEM_outlines(args=sys.argv[1:])
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
