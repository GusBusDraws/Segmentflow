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
                'in_dir_path'  : '01. Input dir path',
                'file_suffix'  : '02. File suffix',
                'slice_crop'   : '03. Slice crop',
                'row_crop'     : '04. Row crop',
                'col_crop'     : '05. Column crop',
                'spatial_res'  : '06. Pixel size',
                'binder_val'   : '07. Intensity of binder voxels',
                'particle_val' : '08. Intensity of particle voxels',
            },
            'B. Segmentation' : {
                'min_peak_distance' :
                    '01. Minimum pixel distance between particle centers',
            },
            'C. Output' : {
                'out_dir_path'     : '01. Path to save output dir',
                'overwrite'        : '02. Overwrite files',
                'out_prefix'       : '03. Output prefix',
                'nslices'          : '04. Number of slices in checkpoint plots',
                'save_checkpoints' : '05. Save checkpoint figures',
                'save_stl'         : '06. Save STL file',
                'save_voxels'      : '07. Save voxel TIF stack'
            },
        }
        self.default_values = {
            'in_dir_path'       : 'REQUIRED',
            'file_suffix'       : '.tif',
            'slice_crop'        : None,
            'row_crop'          : None,
            'col_crop'          : None,
            'spatial_res'       : 1,
            'binder_val'        : 1,
            'particle_val'      : 2,
            'min_peak_distance' : 10,
            'out_dir_path'      : 'REQUIRED',
            'out_prefix'        : '',
            'overwrite'         : True,
            'nslices'           : 5,
            'save_checkpoints'  : True,
            'save_stl'          : True,
            'save_voxels'       : True,
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

    #~~~~~~~~~~#
    # Workflow #
    #~~~~~~~~~~#
    def run(self):
        """Carry out workflow WORKFLOW_NAME as described by WORKFLOW_DESCRIPTION.
        ----------
        Parameters
        ----------
        ui : dict
            Dictionary of inputs loaded from YAML and processed by
            segment.process_args() passed after "-i" flag when running this script.
        """
        #-------------#
        # Load images #
        #-------------#
        print('pre load_inputs():')
        print(f'{self.yaml_path=}')
        # Load YAML inputs into a dictionary
        ui = segment.load_inputs(
            self.yaml_path,
            self.categorized_input_shorthands,
            self.default_values
        )
        print()
        imgs_semantic = segment.load_images(
            ui['in_dir_path'],
            slice_crop=ui['slice_crop'],
            row_crop=ui['row_crop'],
            col_crop=ui['col_crop'],
            file_suffix=ui['file_suffix']
        )
        imgs_semantic = imgs_semantic.astype(int)
        imgs_semantic[imgs_semantic == ui['binder_val']] = 1
        imgs_semantic[imgs_semantic == ui['particle_val']] = 2
        n_fig_digits = 2
        fig_n = 0
        fig, axes = view.vol_slices(
            imgs_semantic,
            nslices=ui['nslices'],
            fig_w=7.5,
            dpi=300
        )
        if ui['save_checkpoints'] == 'show':
            plt.show()
        elif ui['save_checkpoints'] == True:
            plt.savefig(
                Path(ui['out_dir_path'])
                / f'{str(fig_n).zfill(n_fig_digits)}-semantic-seg-imgs.png')

        #-------------------#
        # Segment particles #
        #-------------------#
        print()
        imgs_labeled = segment.watershed_segment(
            imgs_semantic==2,
            min_peak_distance=ui['min_peak_distance'],
            exclude_borders=False,
            return_dict=False
        )
        fig, axes = view.plot_color_labels(
            imgs_labeled,
            nslices=ui['nslices'],
            fig_w=7.5,
            dpi=300
        )
        fig_n += 1
        if ui['save_checkpoints'] == 'show':
            plt.show()
        elif ui['save_checkpoints'] == True:
            plt.savefig(
                Path(ui['out_dir_path'])
                / f'{str(fig_n).zfill(n_fig_digits)}-instance-seg-imgs.png')

        #--------------#
        # Save outputs #
        #--------------#
        # Save STLs
        print()
        if ui['save_stl']:
            segment.save_as_stl_files(
                imgs_labeled,
                ui['out_dir_path'],
                ui['out_prefix'],
                make_new_save_dir=True,
                spatial_res=ui['spatial_res'],
                stl_overwrite=ui['overwrite']
            )
        # Save images
        if ui['save_voxels']:
            segment.save_images(
                imgs_labeled,
                Path(ui['out_dir_path']) / f"{ui['out_prefix']}_labeled_voxels"
            )


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

