import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view
import sys


class Workflow():
    def __init__(self, yaml_path=None, args=None):
        self.name = Path(__file__).stem
        self.description = (
            'This workflow segments an image stack in which all particles are'
            ' labeled with the integer 2. Outputs can be instance-labeled TIF'
            ' stack and/or STL files. Developed for Segmentflow v0.0.3.'
        )
        self.categorized_input_shorthands = {
            'A. Input' : {
                'in_dir_path' : '01. Input dir path to labeled voxels',
                'file_suffix' : '02. File Suffix',
                'slice_crop'  : '03. Slice Crop',
                'row_crop'    : '04. Row Crop',
                'col_crop'    : '05. Column Crop',
            },
            'B: Output' : {
                'out_dir_path' : '01. Output dir path',
                'out_prefix'   : '02. Output prefix',
            },
        }
        self.default_values = {
            'in_dir_path'  : 'REQUIRED',
            'file_suffix'  : '.tif',
            'slice_crop'   : None,
            'row_crop'     : None,
            'col_crop'     : None,
            'out_dir_path' : 'REQUIRED',
            'out_prefix'   : '',
        }
        # A Workflow object has to have some way of loading info/knowing what
        # to do, either with a yaml_path directly (used for testing) or args
        # (this is how a YAML file path is passed from the command line)
        self.yaml_path = None
        if yaml_path is None and args is None:
            raise ValueError(
                'Workflow must be intitialized with either yaml_path or args.')
        elif yaml_path is not None:
            self.yaml_path = Path(yaml_path).resolve()
        else:
            self.yaml_path = Path(self.process_args(args)).resolve()

    def process_args(self, argv):
        # Get command-line arguments
        yaml_path = ''
        if len(argv) == 0:
            help(self.name, self.desc)
            sys.exit()
        if argv[0] == '-g':
            if len(argv) == 2:
                segment.generate_input_file(
                    argv[1],
                    self.name,
                    self.categorized_input_shorthands,
                    self.default_values
                )
            else:
                raise ValueError(
                    'To generate an input file, pass the path of a directory'
                    ' to save the file.'
                )
            sys.exit()
        elif argv[0] == '-h':
            help(self.name, self.desc)
            sys.exit()
        elif argv[0] == "-i" and len(argv) == 2:
            yaml_path = argv[1]
        if yaml_path == '':
            raise ValueError(
                f'No input file specified.'
                f' Enter "python -m segmentflow.workflow.{self.name} -h"'
                f' for more help'
            )
        return yaml_path

    def run(self):
        """Carry out workflow WORKFLOW_NAME as described by WORKFLOW_DESCRIPTION.
        ----------
        Parameters
        ----------
        ui : dict
            Dictionary of inputs loaded from YAML and processed by
            segment.process_args() passed after "-i" flag when running this script.
        """
        #----------------------#
        # Read YAML input file #
        #----------------------#
        # ui = segment.process_args(argv, WORKFLOW_NAME, WORKFLOW_DESCRIPTION)
        ui = {
            'in_dir_path' : 'c:/Users/gusb/Research/mhe-analysis/data/F63tiff',
            'slice_crop' : None,
            'row_crop' : None,
            'col_crop' : None,
            'file_suffix' : '.tif',
        }
        show_checkpoints = False
        checkpoint_save_dir = ui['out_dir_path']

        #-------------#
        # Load images #
        #-------------#
        print()
        imgs_labeled = segment.load_images(
            ui['in_dir_path'],
            slice_crop=ui['slice_crop'],
            row_crop=ui['row_crop'],
            col_crop=ui['col_crop'],
            convert_to_float=False,
            file_suffix=ui['file_suffix']
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
        plt.show()

if __name__ == '__main__':
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('Welcome to Segmentflow!')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    workflow = Workflow(args=sys.argv[1:])
    print(f'Beginning workflow: {workflow.name}')
    print()
    print('pre workflow.run():')
    print(f'{workflow.yaml_path=}')
    workflow.run()
    print()
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('Successful Completion')
    print('~~~~~~~~~~~~~~~~~~~~~')
    print()
