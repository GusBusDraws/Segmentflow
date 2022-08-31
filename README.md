# Segmentflow
Developed by C. Gus Becker (GitHub/GitLab: @cgusb).

This project establishes a segmentation workflow for 3D image data obtained from processes like x-ray computed tomography (CT). Segmented data can be expressed as integer-labeled images (integer value of pixels correspond to unique particles) or separate STL files.

## Contents
1. [Requirements](#requirements)
2. [Regression Testing](#regression)
2. [segment.py](#segment.py)
3. [example-ct-to-stl.ipynb](#example-ct-to-stl.ipynb)
4. [example-single-particle.ipynb](#example-single-particle.ipynb)

## Requirements <a name="requirements"></a>
- Python >= 3.5
- imageio >= 2.21.0
- matplotlib >= 3.5.2
- numpy >= 1.23.1
- numpy-stl >= 2.17.1
- open3d >= 0.15.1
- pandas >= 1.4.3
- PyYAML >= 6.0
- scikit-image >= 0.19.3
- scipy >= 1.9.0

## Regression Testing <a name="regression"></a>
- Before any commit or merge to main, be sure segment.py passes the regression tests.
- To run the regression tests, enter the command in the next bullet, while at the top level of the repository.
```
python ./testing/python/runTests.py -f ./testing/manage/regression.yml
```

## [segment.py](segment.py) <a name="segment.py"></a>
Module containing segmentation workflow functions. Process is split into six steps: input loading, preprocessing, binarization, segmentation, surface meshing, and mesh postprocessing.

### Input Loading
All inputs are stored in a separate YAML file for ease of use and reproducibility. With inputs stored in a separate file, the input used in one run can be reused or slightly altered in future run, while also keeping a record of what may or may not have worked in previous runs. A loading function parses the parameters into a dictionary. Any blank parameters are replaced with default values stored in the loading function. After loading, a copy of the YAML inputs are saved in another YAML file in the STL output directory specified. This reduces the likelihood that the file will be edited while also back-filling default values for parameter values not provided in the initial YAML.

### Preprocessing
Image preprocessing steps include median filter application and intensity clipping. Applying a median filter to the data reduces noise retaining edges (unlike Gaussian filtering which will blur edges). Intesity clipping happens by setting an upper and lower threshold define by intensity percentile and rescaling the data to that range. Each clipped intensity is replaced by the value at the bounds of the clip.

### Binarization
### Segmentation
### Surface Meshing
### Mesh Postprocessing

## [example-ct-to-stl.ipynb](example-ct-to-stl.ipynb) <a name="example-ct-to-stl.ipynb"></a>
General workflow example for entire segmentation process. Process inputs from YAML file to load F50 sand [sample F63](https://micromorph.gitlab.io/projectwebsite/ExpDetailsForSample_F63.html) between slices 300 and 650. This represents the majority of the sample between non-level top and bottom boundaries. After loading, images are preprocessed to reduce noise and improve contrast. Images are binarized using automatic Otsu thresholding to separate the images into three classes: ideally the void, binder, and particles. Only the topmost threshold is used to calculate the binary image. Particles are then segmented using watershed segmentation seeded with the local maxima from the distance map, with local maxima no closer than 7 pixels. Voxels of the segmented particles are converted to a triangular mesh using a marching cubes algorithm and saved as a separate STL file. STL files are then postprocessed to smooth the blocky meshes and simplify the number of triangles to reduce complexity.

## [example-single-particle.ipynb](example-single-particle.ipynb) <a name="example-single-particle.ipynb"></a>
Workflow example of loading a specific particle from aa cropped view of F50 sand [sample F63](https://micromorph.gitlab.io/projectwebsite/ExpDetailsForSample_F63.html), preprocessing, binarizing, and segmenting the particles within the cropped region as in the full example. After segmentation, each of the unique particleID labels are shown overlaid on each particle, a particle is chosen by selecting its label, then a triangular mesh is created for that particle only. The individiaul tri-mesh is saved as an STL file along with its smoothed and simplified versions.

