from pathlib import Path
import yaml
from segmentflow import segment
import shutil


TEMP_PATH = Path('tests/temp')
OUTPUT_PATH = TEMP_PATH / 'output'
YAML_PATH = TEMP_PATH / 'test_input.yml'
YAML_DATA = {
    'Files' : {
        'CT Scan Dir' : str(TEMP_PATH),
        'STL Dir' : str(OUTPUT_PATH),
        'STL Prefix' : None,
        'Particle ID' : None,
        'Overwrite Existing STL Files' : None,
        'Suppress Save Messages' : None,
    },
    'Load' : {
        'File Suffix' : None,
        'Slice Crop' : None,
        'Row Crop' : None,
        'Col Crop' : None,
    },
    'Preprocess' : {
        'Apply Median Filter' : None,
        'Rescale Intensity Range' : None,
    },
    'Binarize' : {
        'Number of Otsu Classes' : None,
        'Number of Classes to Select' : None,
        'Save Isolated Classes' : None,
    },
    'Segment' : {
        'Perform Segmentation' : None,
        'Use Integer Distance Map' : None,
        'Min Peak Distance' : None,
        'Exclude Border Particles' : None,
    },
    'STL' : {
        'Create STL Files' : None,
        'Number of Pre-Surface Meshing Erosions' : None,
        'Smooth Voxels with Median Filtering' : None,
        'Marching Cubes Voxel Step Size' : None,
        'Pixel-to-Length Ratio' : None,
        'Number of Smoothing Iterations' : None,
        'Target number of Triangles/Faces' : None,
        'Simplification factor Per Iteration' : None,
    },
    'Plot' : {
        'Segmentation Plot Create Figure' : None,
        'Segmentation Plot Number of Images' : None,
        'Segmentation Plot Slices' : None,
        'Segmentation Plot Show Maxima' : None,
        'Particle Labels Plot Create Figure' : None,
        'Particle Labels Plot Image Index' : None,
        'STL Plot Create Figure' : None,
    },
}

def test_create_temp():
    TEMP_PATH.mkdir()
    assert TEMP_PATH.exists()

def test_create_input_file():
    with open(str(YAML_PATH), 'w') as file:
        yaml.dump(YAML_DATA, file)
    assert YAML_PATH.exists()

def test_load_input():
    input = segment.load_inputs(YAML_PATH)
    nested_keys = []
    for key in YAML_DATA.keys():
        for nested_key in YAML_DATA[key]:
            nested_keys.append(nested_key)
    assert len(input.keys()) == len(nested_keys)

def test_output_dir():
    assert OUTPUT_PATH.exists()

def test_delete_temp():
    shutil.rmtree(TEMP_PATH)
    assert not TEMP_PATH.exists()

if __name__ == '__main__':
    test_create_temp()
    test_create_input_file()
    input = segment.load_inputs(YAML_PATH)
    nested_keys = []
    print(input.keys())
    for key in YAML_DATA.keys():
        for nested_key in YAML_DATA[key]:
            nested_keys.append(nested_key)
    test_delete_temp()

