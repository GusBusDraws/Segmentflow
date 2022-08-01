# Segmentflow
Developed by C. Gus Becker (GitHub/GitLab: @cgusb).

This project establishes a segmentation workflow for 3D image data obtained from processes like x-ray computed tomography. Segmented data can be expressed as integer-labeled images (integer value of pixels correspond to unique particles) or separate STL files.

## Contents
1. [Requirements](#requirements)
2. [Regression Testing](#regression)
2. [segment.py](#segment.py)
3. [example-ct-to-stl.ipynb](#example-ct-to-stl.ipynb)
4. [example-single-particle.ipynb](#example-single-particle.ipynb)

### Requirements <a name="requirements"></a>
- minimum python3.5
- PyYaml module

### Regression Testing <a name="regression"></a>
- Before any commit or merge to main, be sure segment.py passes the regression tests.
- To run the regression tests, enter the command in the next bullet, while at the top level of the repository.
- ./testing/python/runTests.py -f ./testing/manage/regression.yml

### segment.py <a name="segment.py"></a>
Module containing segmentation workflow functions:
<p align="center">
	[segment.py](segment.py)
</p>

### example-ct-to-stl.ipynb <a name="example-ct-to-stl.ipynb"></a>
<p align="center">
	[example-ct-to-stl.ipynb](example-ct-to-stl.ipynb)
</p>
General workflow example for entire segmentation process. Load a cropped region of an F50 sand sample (PetaLibrary/CT Scans/Sand Compression in Steps/NoComptiff/), segment each particle, convert each particle to a triangular mesh, and save each tri-mesh as a separate STL file.

### example-single-particle.ipynb <a name="example-single-particle.ipynb"></a>
<p align="center">
	[example-single-particle.ipynb](example-single-particle.ipynb)
</p>
Workflow example of loading a cropped region of the same F50 sand sample (PetaLibrary/CT Scans/Sand Compression in Steps/NoComptiff/), segmenting each particle, showing the labels of each particle, picking a particle by its label, creating a triangular mesh for the particle using a marching cubes algorithm, and saving the individiaul tri-mesh as an STL file.

