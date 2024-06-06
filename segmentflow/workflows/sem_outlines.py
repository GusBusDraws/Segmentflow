import imageio.v3 as iio
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from segmentflow import segment, view
from segmentflow.workflows.workflow import Workflow
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
import sys


class Labels_to_size(Workflow):
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
                'perform_semantic'  : '05. Perform semantic segmentation',
                'thresh_vals'       : '06. Threshold values',
                'perform_instance'  : '07. Perform instance segmentation',
                'merge_aid'         : '08. Open merging aid figure',
                'merge_path'        : '09. Path to TXT file defining merges'
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
            'perform_semantic' : False,
            'thresh_vals'      : None,
            'perform_instance' : False,
            'merge_aid'        : True,
            'merge_path'       : 'REQUIRED',
            'out_dir_path'     : 'REQUIRED',
            'out_prefix'       : '',
            'overwrite'        : False,
        }

    def run(self):
        """Carry out workflow WORKFLOW_NAME as described by
        WORKFLOW_DESCRIPTION.
        """
        show_checkpoints = False
        checkpoint_save_dir = self.ui['out_dir_path']

        #------------#
        # Load image #
        #------------#
        print()
        img_dir_path = self.ui['sem_path']
        img = iio.imread(img_dir_path)
        print(f'Image loaded: {img.shape}')
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
        hist, bins_edges = np.histogram(img_crop, bins=256)
        img_labeled = segment.isolate_classes(
            img_crop, self.ui['thresh_vals'], intensity_step=1)
        n_pixels = img_labeled.shape[0] * img_labeled.shape[1]
        n_void = np.count_nonzero(img_labeled == 0)
        n_binder = np.count_nonzero(img_labeled == 1)
        n_crystals = np.count_nonzero(img_labeled == 2)
        print('--> Void area fraction:', n_void / n_pixels)
        print('--> Crystal area fraction:', n_crystals / n_pixels)
        print(
            '--> Crystal area fraction (void corrected):',
            n_crystals / (n_pixels - n_void))
        # Figure: Semantic segmentation
        fig, axes = plt.subplots(
            1, 3, dpi=300, facecolor='white', figsize=(9, 3),
            constrained_layout=True)
        axes[0].imshow(img_crop, vmin=img.min(), vmax=img.max(), cmap='gray')
        axes[0].set_axis_off()
        axes[0].set_title('Subarea')
        axes[1].plot(bins_edges[:-1], hist, c='red', zorder=1)
        colors = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=0, vmax=2)
        span_vals = [0] + self.ui['thresh_vals'] + [2**16]
        for i in range(0, len(span_vals)-1):
            axes[1].axvspan(
                span_vals[i], span_vals[i + 1], facecolor=colors(norm(i)),
                zorder=0)
        axes[1].set_xlim([0, 2**16])
        axes[1].set_aspect(2**16/400)
        axes[1].set_ylabel('Counts')
        axes[1].set_xlabel('Pixel Intensities')
        axes[1].set_title('Histogram')
        axes[2].imshow(img_labeled)
        axes[2].set_axis_off()
        axes[2].set_title('Isolated Classes')
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='semantic-seg')

        #-----------------------#
        # Instance segmentation #
        #-----------------------#
        # Segment the grains with a watershed algorithm
        if self.ui['perform_instance']:
            img_labeled = segment.watershed_segment(
                img_labeled==2, min_peak_distance=2)
            img_colors = view.color_labels(img_labeled, return_image=True)
            # Apply manual segmentation adjustment before clearing border
            img_labeled[0, 18] = 0
            img_labeled[0, 21] = 0
            img_labeled[0, 22] = 0
            img_labeled[1, 22] = 0
            img_labeled[0, 37:46] = 0
            # Clear segmentation border
            cleared_labeled = segmentation.clear_border(img_labeled)
            cleared_colors = view.color_labels(
                cleared_labeled, return_image=True)
            # Figure: Initial instance segmentation
            fig, axes = view.images(
                [img_crop, img_colors, cleared_colors],
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
                [img_crop, cleared_colors, cleared_labeled],
                imgs_per_row=3, dpi=300)

        #---------------#
        # Merge regions #
        #---------------#
        # Merge regions grouped by line in a TXT file
        merge_groups = [
            [15, 18, 29, 30, 32, 33, 35],
            [16, 19],
            [47, 51, 54],
            [57, 62],
            [67, 68, 72],
            [82, 83, 84, 89],
            [59, 65],
            [70, 71],
            [80, 88],
            [104, 105, 109]
        ]
        merge_labeled = cleared_labeled.copy()
        for merge in merge_groups:
            for label in merge:
                merge_labeled[cleared_labeled == label] = merge[0]
        merge_colors = view.color_labels(merge_labeled, return_image=True)
        # Figure: Brief figure showing merged regions
        fig, axes = view.images(
            [
                img_crop, cleared_colors, segmentation.mark_boundaries(
                    merge_colors, merge_labeled, mode='subpixel',
                    color=(1,1,1)
                ),
            ], imgs_per_row=3, dpi=300, subplot_letters=True, fig_w=8
        )
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='merged-regions-detailed')
        # Figure: Detailed figure showing merged regions
        fig, axes = view.images(
            [
                img_crop,
                img_labeled,
                img_colors,
                cleared_colors,
                merge_colors,
                segmentation.mark_boundaries(
                    merge_colors, merge_labeled, mode='subpixel',
                    color=(1,1,1)
                ),
            ], imgs_per_row=3, dpi=300, subplot_letters=True, fig_w=8
        )
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='merged-regions-detailed')

        #-----------------------------#
        # Select bounding coordinates #
        #-----------------------------#
        # Select the bounding coordinates of each segmented grain
        labels = np.unique(merge_labeled)
        # Difference between region & eroded region
        merge_bounds = np.zeros_like(merge_labeled)
        # Overlap between bounds and eroded -> dilated bounds
        merge_clean = np.zeros_like(merge_labeled)
        for i in labels[labels > 0]:
            # Isolate/binarize region label
            reg_bin = np.zeros_like(merge_labeled)
            reg_bin[merge_labeled == i] = 1
            reg_bin = ndi.binary_fill_holes(reg_bin).astype(np.ubyte)
            # Create bounds by subtracting eroded region
            reg_bounds = reg_bin - morphology.binary_erosion(reg_bin)
            # Set resulting bounds to original label in collection
            merge_bounds[reg_bounds == 1] = i
            # Ensure the bounding coordinates can create a loop
            reg_bin_ero_dil = morphology.binary_erosion(reg_bin)
            reg_bin_ero_dil = morphology.binary_dilation(reg_bin_ero_dil)
            merge_clean[(reg_bounds == 1) & (reg_bin_ero_dil == 1)] = i
        # Figure: Initial instance segmentation
        fig, axes = view.images([
            view.color_labels(merge_labeled, return_image=True),
            view.color_labels(merge_bounds, return_image=True),
            view.color_labels(merge_clean, return_image=True),
        ], imgs_per_row=3, dpi=200, subplot_letters=True)
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='region-bounds')

        #---------------------------#
        # Save bounding coordinates #
        #---------------------------#
        # Save the resulting ordered coordinates in CSV files (one per grain)
        bounding_loops_dir_path = (
            Path(self.ui['out_dir_path']) / f"{self.ui['out_prefix']}_bounding_loops")
        if not bounding_loops_dir_path.exists():
            bounding_loops_dir_path.mkdir(parents=True)
        regions = measure.regionprops(merge_clean)
        n_digits = len(str(len(regions)))
        for region in regions:
            coords = region.coords
            # Compute centroid
            cent=(sum(coords[:, 1])/len(coords),sum(coords[:, 0])/len(coords))
            # Sort by polar angle
            ordered = np.array(
                sorted(
                    coords,
                    key=lambda p: math.atan2(p[0]-cent[1], p[1]-cent[0])
                )
            )
            ordered_loop = np.zeros((ordered.shape[0] + 1, 2))
            ordered_loop[-1, :] = ordered[0, :]
            ordered_loop[:-1, :] = ordered
            x = ordered_loop[:, 1]
            y = ordered_loop[:, 0]
            df = pd.DataFrame(data={'x': x, 'y': y})
            df.to_csv(
                bounding_loops_dir_path
                / f'{str(region.label).zfill(n_digits)}.csv'
            )
        # Save a single CSV file containing the bounding boxes of all grains
        region_table = measure.regionprops_table(
            merge_clean, properties=('label', 'bbox'))
        bbox_df = pd.DataFrame(region_table)
        bbox_df.rename(columns={
            'bbox-0': 'min_row',
            'bbox-1': 'min_col',
            'bbox-2': 'max_row',
            'bbox-3': 'max_col',
        }, inplace=True)
        bbox_df['ums_per_pixel'] = [self.ui['spatial_res']] * bbox_df.index.shape[0]
        bbox_df.to_csv(
            Path(self.ui['out_dir_path']) / f"{self.ui['out_prefix']}_bounding_boxes.csv")

if __name__ == '__main__':
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    workflow = Workflow(args=sys.argv[1:])
    print(f'Beginning workflow: {workflow.name}')
    print()
    workflow.run()
    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Successful Completion')
    print('~~~~~~~~~~~~~~~~~~~~~')
    print()
