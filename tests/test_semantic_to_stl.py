import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pytest
from segmentflow import segment, view
from segmentflow.workflows import semantic_to_stl
import shutil
from skimage import draw, measure
import yaml


class TestClass():
    temp_path = Path('tests/temp')
    yaml_path = Path(temp_path) / 'input.yml'
    cubes_path = Path(temp_path) / 'cubes'
    sf_prefix = 'test_cubes'
    sf_path = Path(temp_path) / sf_prefix
    duplicate_yaml_path = (
        Path(sf_path) / f'{sf_prefix}_input.yml')
    stl_path = Path(sf_path) / f'{sf_prefix}_STLs'
    props_path = (
        Path(stl_path) / f'{sf_prefix}_properties.csv')


    def test_make_temp_dir(self):
        # Create temp dir
        self.temp_path.mkdir()
        assert self.temp_path.exists()

    def test_make_cubes(self):
        grid_w = 70
        cube_w = 10
        cube_spacing = 10
        cubes = np.zeros((grid_w, grid_w, grid_w), dtype=np.ubyte)
        for i in np.arange(cube_w, grid_w, cube_w + cube_spacing):
            for r in np.arange(cube_w, grid_w, cube_w + cube_spacing):
                for c in np.arange(cube_w, grid_w, cube_w + cube_spacing):
                    i_inds, r_inds, c_inds = draw.rectangle(
                        start=(i, r, c), extent=(cube_w, cube_w, cube_w)
                    )
                    cubes[i_inds, r_inds, c_inds] = 255
        # fig, axes = view.vol_slices(cubes, nslices=7)
        # plt.show()
        segment.save_images(cubes, self.cubes_path)
        assert self.cubes_path.exists()

    def test_num_cubes(self):
        cubes = segment.load_images(self.cubes_path, file_suffix='.tif')
        cubes_labeled = measure.label(cubes)
        # Subtract 1 to account for 0 label
        n_cubes = len(np.unique(cubes_labeled)) - 1
        assert n_cubes == 27

    def test_make_yaml(self):
        inputs = {
            'A. Input' : {
                '01. Input dir path' : str(self.cubes_path.resolve()),
                '02. File suffix' : '.tif',
                '03. Slice crop' : None,
                '04. Row crop' : None,
                '05. Column crop' : None,
                '06. Pixel size' : 1,
                '07. Intensity of binder voxels' : 1,
                '08. Intensity of particle voxels' : 255,
            },
            'B. Segmentation' : {
                '01. Minimum pixel distance between particle centers' : 10,
            },
            'C. Output' : {
                '01. Path to save output dir' : str(self.sf_path.resolve()),
                '02. Overwrite files' : False,
                '03. Output prefix' : self.sf_prefix,
                '04. Number of slices in checkpoint plots' : 7,
                '05. Save checkpoint figures' : True,
                '06. Save STL file' : True,
                '07. Save voxel TIF stack' : True,
            },
        }
        with open(str(self.yaml_path), 'w') as file:
            doc = yaml.dump(inputs, file, sort_keys=False)
        assert self.yaml_path.exists()

    def test_yaml_contents(self):
        # Check YAML contents to match workflow.categorized_input_shorthand vals
        pass

    def test_call_sf(self):
        try:
            workflow = semantic_to_stl.Workflow(yaml_path=self.yaml_path)
            workflow.run()
        except Exception as e:
            pytest.fail('An unexpected error occurred.')

    def test_output_dir(self):
        assert self.sf_path.exists()

    def test_duplicate_yaml(self):
        assert self.duplicate_yaml_path.exists()

    def test_stl_dir(self):
        assert self.stl_path.exists()

    def test_props(self):
        assert self.props_path.exists()

    def test_num_stls(self):
        stl_path_list = [path for path in self.stl_path.glob('*.stl')]
        assert len(stl_path_list) == 27

    def test_rm_temp_dir(self):
        shutil.rmtree(self.temp_path)
        assert not self.temp_path.exists()


if __name__ == '__main__':
    test = TestClass()
    test.test_make_temp_dir()
    test.test_make_yaml()
    test.test_make_cubes()
    test.test_num_cubes()
    test.test_call_sf()
    test.test_output_dir()
    test.test_stl_dir()
    test.test_props()
    test.test_num_stls()
    test.test_rm_temp_dir()
    print('All tests passed!')

