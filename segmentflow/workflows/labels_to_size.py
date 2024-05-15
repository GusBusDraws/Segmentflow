from pathlib import Path
from segmentflow import segment, view
from segmentflow.workflows.workflow import Workflow
import sys


class Labels_to_size(Workflow):
    def __init__(self, yaml_path=None, args=None):
        # Initialize parent class to set yaml_path
        super().__init__(yaml_path=yaml_path, args=args)
        self.name = Path(__file__).stem
        self.description = (
            'This workflow takes a segmented and labeled image stack and'
            ' calculates the size distribution of the particles.'
            ' Developed for Segmentflow v0.0.4.'
        )
        self.categorized_input_shorthands = {
            'A. Input' : {
                'in_dir_path' : '01. Input dir path to labeled voxels',
                'file_suffix' : '02. File Suffix',
                'slice_crop'  : '03. Slice Crop',
                'row_crop'    : '04. Row Crop',
                'col_crop'    : '05. Column Crop',
                'spatial_res' : '06. Pixel size',
                'material'    : '07. Material system',
            },
            'B. Output' : {
                'out_dir_path' : '01. Output dir path',
                'out_prefix'   : '02. Output prefix',
                'overwrite'    : '03. Overwrite files',
            },
        }
        self.default_values = {
            'in_dir_path'  : 'REQUIRED',
            'file_suffix'  : '.tif',
            'slice_crop'   : None,
            'row_crop'     : None,
            'col_crop'     : None,
            'spatial_res'  : 1,
            'material'     : 'IDOX',
            'out_dir_path' : 'REQUIRED',
            'out_prefix'   : '',
            'overwrite'    : False,
        }

    def run(self):
        """Carry out workflow (self.name) as described by self.description.
        """
        show_checkpoints = False
        checkpoint_save_dir = self.ui['out_dir_path']

        #-------------#
        # Load images #
        #-------------#
        self.logger.info(f'Beginning workflow: {workflow.name}')
        imgs_labeled = segment.load_images(
            self.ui['in_dir_path'],
            slice_crop=self.ui['slice_crop'],
            row_crop=self.ui['row_crop'],
            col_crop=self.ui['col_crop'],
            convert_to_float=False,
            file_suffix=self.ui['file_suffix'],
            logger=self.logger
        )
        fig, axes = view.plot_color_labels(
            imgs_labeled, nslices=3, exclude_bounding_slices=True, fig_w=7.5,
            dpi=300
        )
        fig_n = 0
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='labeled-imgs')

        #---------------------------#
        # Analyze size distribution #
        #---------------------------#
        dims_df = segment.get_dims_df(imgs_labeled, logger=self.logger)
        n_particles, sieve_sizes = segment.simulate_sieve_bbox(
            dims_df, self.ui['material'], pixel_res=self.ui['spatial_res'],
            logger=self.logger)
        fig, ax = view.grading_curve(
            n_particles, sieve_sizes, standard=self.ui['material'],
            standard_label=f"{self.ui['material']} standard")
        fig_n += 1
        segment.output_checkpoints(
            fig, show=show_checkpoints, save_path=checkpoint_save_dir,
            fn_n=fig_n, fn_suffix='size-dist')
        segment.save_binned_particles_csv(
            self.ui['out_dir_path'], sieve_sizes, n_particles,
            output_prefix=self.ui['out_prefix'], logger=self.logger)

if __name__ == '__main__':
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    # Pass path to YAML from args and store in workflow class
    workflow = Labels_to_size(args=sys.argv[1:])
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
