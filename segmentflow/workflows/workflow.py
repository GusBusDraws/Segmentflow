import logging
import matplotlib.pyplot as plt
from pathlib import Path
from segmentflow import segment, view
import sys


class Workflow():
    def __init__(self, yaml_path=None, args=None):
        # These properties are updated in child class (i.e. specific workflow)
        self.name = None
        self.description = None
        self.categorized_input_shorthands = None
        self.default_values = None
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

    def read_yaml(self):
        # Read YAML input file
        self.ui = segment.load_inputs(
            self.yaml_path,
            self.categorized_input_shorthands,
            self.default_values
        )

    def create_logger(self):
        # Set path for log file
        log_path = Path(self.ui['out_dir_path']) / 'segmentflow.log'
        # Create a logger object
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        # Create a formatter to define the log format
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        # Create a stream handler to print logs to the console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

