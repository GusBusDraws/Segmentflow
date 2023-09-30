# Segmentflow
<!------------------------------------------------------------------------>
Developed by C. Gus Becker (GitHub/GitLab: @cgusb).

Segmentflow is a Python package that makes it easy to establish image
segmentation workflows, especially for generating voxel and surface mesh
geometries for 3D image data obtained from processes like x-ray computed
tomography (CT). Segmented data can be exported in a variety of formats
including collections of binary images, integer-labeled voxels (integer
value of pixels corresponds to unique particles) or collections of STL files.

## Contents
<!------------------------------------------------------------------------>
1. [Requirements](#requirements)
1. [Typical Routine](#typical-routine)
1. [Regression Testing](#regression-testing)
1. [Notebooks](#notebooks)
1. [Workflow Scripts](#workflow-scripts)
1. [Change log](#change-log)

## Requirements
<!------------------------------------------------------------------------>
Required for install:
- Python >= 3.5
- imageio >= 2.21.0
- matplotlib >= 3.5.2
- NumPy >= 1.23.1
- numpy-stl >= 2.17.1
- pandas >= 1.4.3
- PyYAML >= 6.0
- scikit-image >= 0.19.3
- SciPy >= 1.9.0

Required for `mesh` submodule:
- Open3d >= 0.15.1

## Getting Started
<!------------------------------------------------------------------------>
It's recommended to install Segmentflow as a Python package in editable mode
with pip by cloning the repository, activating a virtual environment,
navigating to the root directory of the repository, and using the command:
```bash
python -m pip install -e path/to/segmentflow
```
This will allow you to pull updates to Segmentflow from GitLab and use the 
updated package without uninstalling and re-installing the package.

There are three ways to run Segmentflow:
1. Use the Segmentflow API to write a Python script or Jupyter notebook,
2. Execute the default `workflow` submodule as a script with a YAML input
   file, or
3. Use a workflow script/input file combination from `./workflows/`.

To execute the a Segmentflow workflow `WORKFLOW_NAME`, execute the workflow
with an input file as follows:
```bash
python -m segmentflow.workflows.WORKFLOW_NAME -i path/to/input.yml
```
Note that each workflow file will have a different input file.

## Typical routine

### Input Loading
<!------------------------------------------------------------------------>
All inputs are stored in a separate YAML file for ease of use and
reproducibility. With inputs stored in a separate file, the input used in
one run can be reused or slightly altered in future run, while also keeping
a record of what may or may not have worked in previous runs. A loading
function parses the parameters into a dictionary. Any blank parameters are
replaced with default values stored in the loading function. After loading,
a copy of the YAML inputs are saved in another YAML file in the STL output
directory specified. This reduces the likelihood that the file will be
edited while also back-filling default values for parameter values not
provided in the initial YAML.

### Preprocessing
<!------------------------------------------------------------------------>
Image preprocessing steps include median filter application and intensity
clipping. Applying a median filter to the data reduces noise retaining
edges (unlike Gaussian filtering which will blur edges). Intensity
clipping happens by setting an upper and lower threshold define by
intensity percentile and rescaling the data to that range. Each clipped
intensity is replaced by the value at the bounds of the clip.

### Binarization
<!------------------------------------------------------------------------>
Image binarization is performed by applying a multi-Otsu threshold
algorithm to generate threshold values which divide an image into N
regions. This is done by maximizing inter-class variance.

### Segmentation
<!------------------------------------------------------------------------>
Image segmentation is performed by calculating a distance map from the
binary images which maps the distance to the nearest background pixel to
each foreground pixel. Local maxima are calculated based on a minimum
distance (aligned with minimum particle size) and are used to seed a
watershed segmentation which "floods" the inverted distance map starting
with the seed points as "pouring locations". Can also be thought of as
growing outwards from the seed points. Result of segmentation is a series
of images representing the same volume loaded with the input slice, row,
and column crops. Pixels in the segmented region are 0 for background and
an integer ID if the pixel belongs to a segmented particle. Each particle
has a unique ID ranging from 1 to N particles segmented.

### Surface Meshing
<!------------------------------------------------------------------------>
Surface meshes are created for the segmented particles using a marching
cubes algorithm implemented in scikit-image. There are some voxel
processing methods available before surface meshing is performed such
as voxel smoothing and particle erosions. In voxel smoothing, a median
filter is applied to the voxels of an isolated, binarized particle such
that each voxel is replaced by the median value (0 or 1) of the surrounding
26 voxels (3x3x3 cube). The voxels can also be subject to a series of
morphologic erosions, in which the outer layer of voxels is removed,
similar to the peeling of an onion. After these steps, the marching cubes
algorithm is applied with a specified voxel step size which determines the
granularity of the surface mesh.
The surface meshes output from the marching cubes algorithm are blocky
due to the limited number of surface normals output from the data. Normals
can take the form of each of the 6 Cartesian vectors for voxels on a flat
surface of the particles, as well as the 12 vectors halfway between
each of the Cartesian directions for voxels on the corners, and the 8
vectors between each set of three connecting edges for the corner voxels.

### Mesh Postprocessing
<!------------------------------------------------------------------------>
Mesh postprocessing steps consist of either Laplacian smoothing of the
mesh and/or mesh simplification to reduce the number of triangles/surface
elements. Smoothing the blocky surface meshes output by the marching cubes
algorithm can result in meshes that are more similar to the particles in
reality. These meshes may still have a large number of surface elements
however because the smoothing operation does not change the number of
triangles. To reduce the number of triangles, simplification can be
performed by providing a target number of triangles to scale down the mesh.
This can be done in a single step, or by iteratively reducing the number
of triangles by a specified factor.

### Outputs
<!------------------------------------------------------------------------>
Segmentflow outputs STL files for each particle segmented according to the
provided input parameters. In addition to these STL files, a copy of the
input parameter YAML file (with blank values backfilled with default values)
and a properties CSV file that includes details about each particle.
Currently, the properties CSV includes:

- ID
- Number of voxels
- Centroid (x, y, z)
- Minimum slice bounds
- Maximum slice bounds
- Minimum row bounds
- Maximum row bounds
- Minimum column bounds
- Maximum column bounds

[Back to top](#segmentflow)

## Regression Testing
<!------------------------------------------------------------------------>
- Before any commit or merge to main, be sure segment.py passes the
regression tests.
- To run the regression tests, enter the command in the next bullet, while
at the top level of the repository.
```
python ./testing/python/runTests.py -f ./testing/manage/regression.yml
```

## Notebooks

### [example-CT-to-STL.ipynb](notebooks/example-ct-to-stl.ipynb)
<!------------------------------------------------------------------------>
General workflow example for entire segmentation process. Process inputs
from YAML file to load F50 sand [sample F63](
    https://micromorph.gitlab.io/projectwebsite/ExpDetailsForSample_F63.html)
between slices 300 and 650. This represents the majority of the sample
between non-level top and bottom boundaries. After loading, images are
preprocessed to reduce noise and improve contrast. Images are binarized
using automatic Otsu thresholding to separate the images into three
classes: ideally the void, binder, and particles. Only the topmost
threshold is used to calculate the binary image. Particles are then
segmented using watershed segmentation seeded with the local maxima from
the distance map, with local maxima no closer than 7 pixels. Voxels of
the segmented particles are converted to a triangular mesh using a
marching cubes algorithm and saved as a separate STL file. STL files are
then postprocessed to smooth the blocky meshes and simplify the number of
triangles to reduce complexity.

### [example-single-particle.ipynb](notebooks/example-single-particle.ipynb)
<!------------------------------------------------------------------------>
Workflow example of loading a specific particle from a cropped view of
F50 sand [sample F63](
    https://micromorph.gitlab.io/projectwebsite/ExpDetailsForSample_F63.html),
preprocessing, binarizing, and segmenting the particles within the
cropped region as in the full example. After segmentation, each of the
unique particleID labels are shown overlaid on each particle, a particle
is chosen by selecting its label, then a triangular mesh is created for
that particle only. The individual tri-mesh is saved as an STL file
along with its smoothed and simplified versions.

[Back to top](#segmentflow)

## Workflow Scripts

### [F67_segment_stl.py](segmentflow/workflows/F67_segment_stl.py)

Input parameters are described below with object types beside the
parameter name (str: string, int: integer, float: floating point value,
bool: Boolean/True or False) and default values at the end of each
description:

Files :
- Input dir path : str or pathlib.Path
  - The path to the directory containing CT images.
  Required - no default value
- File Suffix : str
  - Suffix of images to load. Defaults to 'tiff'
- Pixel size : float
  - Size of a pixel/voxel in the CT images. Used to set scale in STL files.
  Defaults to 1.
- Slice Crop : list of two int values or None
  - List defining two cropping values in the slice (image number) dimension.
  None will use all values. Defaults to None
- Row Crop : list of two int values or None
  - List defining two cropping values in the row (y) dimension.
  None will use all values. Defaults to None
- Column Crop : list of two int values or None
  - List defining two cropping values in the column (x) dimension.
  None will use all values. Defaults to None
- Path to save output dir : str or pathlib.Path
  - Path specifying directory to which output files (copy of YAML
  parameters, properties.csv, and/or STL files, etc.) will be
  saved. Required - no default value
- STL Prefix : str
  - String prefix which will be appended to the front of output files.
  Defaults to '' (empty string)
- Suppress Save Messages : bool
  - If False, prints a message for each STL saved.
- Overwrite files : bool
  - If True, existing STL files will be overwritten. If False and
  files already exist at specified output directory, ValueError
  will be raised. Defaults to False

View :
  - Slices to view : list
  Image indices used in plots visualizing the volumes.
  - View raw images : bool
  If True, return figures and axes of a plot showing slices of the raw images,
  as loaded. Defaults to True.
  - View preprocessed images : bool
  If True, return figures and axes of a plot showing slices of the
  preprocessed images. Defaults to True.
  - View semantic images : bool
  If True, return figures and axes of a plot showing slices of the
  semantically segmented images. This assigns a different label that
  approximates the different material types in the volume. Defaults to True.
  - View labeled images : bool
  If True, return figures and axes of a plot showing slices of the
  instance segmented images. This assigns a different label for each segmented
  particle. Defaults to True.

Preprocess:
- Apply median filter : True
  - If True, median filter will be applied to images before binarization.
  Defaults to False.
- Rescale intensity range : list of two int values or None
  - List of two percentile values to use to clip the intensity range of the
  images. Defaults to None.

Segmentation :
- Histogram bins for calculating thresholds : int
  - Number of bins to which image histogram will be condensed before
  2D Gaussian smoothing. After this smoothing, peaks will be identified,
  then the minima between the peaks are selected as threshold values.
  Defaults to 256.
- View histogram with threshold values : bool
  If True, figure and axis will be returned to show the
  the threshold values overlaid on the raw and smoothed histogram.
  Defaults to True.
- Upper and lower y-limits of histogram : None or list
  - If list, values 0 and 1 will be used as minimum and maximum to
  rescale y-axis in plot. Defaults to None.
- Perform segmentation : bool
  - If True, segmentation will continue. This parameter gives a breakout
  option if only the binarization is need. Defaults to True.
- Min peak distance : int
  - Minimum distance allowed for distance map peaks or maxima that are used
  to seed watershed segmentation. Can be thought to "grow" into segmented
  regions. Each maxima will correspond to one segmented particle.
  A good way to determine this number is to take the minimum particle
  size (0.075 mm for F50 sand) and divide by spatial resolution (~0.010 mm
  for microfocus x-radiography).
  Defaults to 6
- Exclude border particles : bool
  - If True, particles touching the border of the volume (the edges of the
  3D collection of images) will be removed after segmentation.
  Defaults to False.

STL :
- Create STL files : bool
  - If True, STL files are created post segmentation. Defaults to True
- Suppress save message for each STL file : bool
  - If False, a message is printed to the console for each STL file saved.
  Defaults to False.
- Number of pre-surface meshing erosions : int
  - Number of erosions to perform in succession following segmentation of
  particles. Each erosion can be thought of as peeling off the outer "onion
  skin" of particle voxels. Defaults to 0.
- Smooth Voxels with Median Filtering : bool
  - If True, smooth particles using a median filter before marching cubes
  surface meshing. This filter replaces each voxel of an isolated,
  binarized particle with the median value of it's 26 neighbors
  (3x3x3 cube minus self). Since this is operating on a binary image,
  the median value will be either 0 or 1, so no further thresholding is
  need to turn the particle back into a binary image.
  This has the effect of smoothing out particles jutting out from the
  volumes and filling in holes/divots on the surface.
  Defaults to False.
- Marching cubes voxel step size : int
  - Number of voxels to iterate across surface during marching cubes
  algorithm to create surface mesh. Step size 1 creates highest level
  of detail, with larger integers creating coarser meshes. The result
  is blocky sue to the limited number of surface normals. Normal vectors
  can take the form of each of the 6 Cartesian vectors for voxels on a
  flat surface of the particles, as well as the 12 vectors halfway between
  each of the Cartesian directions for voxels on the corners, and the 8
  vectors between each set of three connecting edges for the corner voxels.
- Number of smoothing iterations : int or None
  - If True, smooth particles following marching cubes surface meshing.
  Surface meshing with the marching cubes algorithm produces blocky
  particles with a limited amount of surface normal directions (simple 6
  Cartesian vectors plus 12 oriented between each pair of connecting faces
  normals and 8 vectors oriented between each triplet of edges).
  Defaults to None.
- Target number of triangles/faces : int or None
  - Desired number of triangles to attempt to reach when simplifying the
  mesh. Mesh will not be simplified if set to None. Defaults to None
- Simplification factor per iteration : int or None
  - Factor by which number of triangles will be reduced in each iteration
  while still above target number of triangles. Setting 2 will reduce
  number of triangles. Defaults to None

[Back to top](#segmentflow)

## Change log
### 0.0.3
- Fix polluted namespace in which `stl.mesh` imported as `mesh`, conflicting with `segmentflow.mesh` imported as `mesh`.
- Added version log to README
- Add workflow script [labels_to_stl.py](segmentflow/workflows/labels_to_stl.py)
- Add `segmentflow.segment.calc_voxel_stats()` for determining binder to particle voxel ratios
- Add workflow script [postprocess_stls.py](segmentflow/workflows/postprocess_stls.py)
- Update `mesh.prostprocess_meshes()` to allow first STL to be skipped (in the case of first STL corresponding to binder).
- Add workflow script [F50_single_grain_segment.py](segmentflow/workflows/F50_single_grain_segment.py)
- Update `view.plot_slices()` to plot last slice when prompted with keyword arg `nslices`
- Add workflow script [semantic_to_stl.py](segmentflow/workflows/semantic_to_stl.py)
- Add STL viewing capability to [segmentflow.view](segmentflow/view.py)
- Add checkpoint images & printed voxel stats to [F83_01_segment.py](segmentflow/workflows/F83_segment.py)
- Rename F83_01_segment.py to [F82_segment.py](segmentflow/workflows/F83_segment.py)
- Return STL vectors from `segment.create_surface_mesh()`
- Add STL min/max to properties.csv saved in `segment.save_as_stl_files()` to verify matching dimensions to input voxels
- Add 'n_voxels_post_erosion' column to properties.csv to quantify volume change following erosion
- Add `color_labels()` as alternative to `plot_color_labels()` and fix image slicing logic
- Wrap checkpoint show/save logic into function `output_checkpoints()`
- Add workflow [IDOX_CHESS.py](segmentflow/workflows/IDOX_CHESS.py)
- Add workflow [instance_to_stl.py](segmentflow/workflows/IDOX_CHESS.py)
- Add `segment.fill_holes()` for filling holes in semantic-segmented images
- Add `segment.fill_holes()` to [IDOX_CHESS](segmentflow/workflows/IDOX_CHESS.py) workflow
- Add print statement for generating histogram in `view.histogram()`
- Update `view.hist()` with ability to mark values on plot
- Update `instance_to_stl` workflow with ability to exclude border particles
- Add output_checkpoints to `IDOX_pours` workflow

[Back to top](#segmentflow)

