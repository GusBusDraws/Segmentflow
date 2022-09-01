# Segmentflow
<!------------------------------------------------------------------------>
Developed by C. Gus Becker (GitHub/GitLab: @cgusb).

This project establishes a segmentation workflow for 3D image data obtained
from processes like x-ray computed tomography (CT). Segmented data can be
expressed as integer-labeled images (integer value of pixels correspond to
unique particles) or separate STL files.

## Contents
<!------------------------------------------------------------------------>
1. [Requirements](#requirements)
2. [Regression Testing](#regression)
2. [segment.py](#segment.py)
3. [example-ct-to-stl.ipynb](#example-ct-to-stl.ipynb)
4. [example-single-particle.ipynb](#example-single-particle.ipynb)

## Requirements <a name="requirements"></a>
<!------------------------------------------------------------------------>
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
<!------------------------------------------------------------------------>
- Before any commit or merge to main, be sure segment.py passes the
regression tests.
- To run the regression tests, enter the command in the next bullet, while
at the top level of the repository.
```
python ./testing/python/runTests.py -f ./testing/manage/regression.yml
```

## [segment.py](segment.py) <a name="segment.py"></a>
<!------------------------------------------------------------------------>
Module containing segmentation workflow functions. Process is split into six
steps: input loading, preprocessing, binarization, segmentation, surface
meshing, and mesh postprocessing.

### Input Loading
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

Input parameters are described below with object types beside the
parameter name (str: string, int: integer, float: floating point value,
bool: boolean/True or False) and default values at the end of each
description:

- Files :
  - **CT Scan Dir** : str or pathlib.Path

    The path to the directory containing CT images.
    Required - no default value

  - STL Dir : str or pathlib.Path

    Path specifying directory to which output files (copy of YAML
    parameters, properties.csv, and/or STL files, etc.) will be
    saved. Required - no default value

  - STL Prefix : str

    String prefix which will be appended to the front of output files.
    Defaults to '' (empty string)

  - Overwrite Existing STL Files : bool

    If True, existing STL files will be overwritten. If False and
    files already exist at specified output directory, ValueError
    will be raised. Defaults to False

  - Suppress Save Messages : bool

    If False, prints a message for each STL saved.

- Load :
  - File Suffix : str

    Suffix of images to load. Defaults to 'tiff'

  - Slice Crop : list of two ints or None

    List defining two cropping values in the slice (image number) dimension.
    None will use all values. Defaults to None

  - Row Crop : list of two ints or None

    List defining two cropping values in the row (y) dimension.
    None will use all values. Defaults to None

  - Col Crop : list of two ints or None

    List defining two cropping values in the column (x) dimension.
    None will use all values. Defaults to None

- Preprocess:
  - Apply Median Filter : True

    If True, median filter will be applied to images before binarization.
    Defaults to False.

  - Rescale Intensity Range : list of two ints or None

    List of two percentile values to use to clip the intesity range of the
    images. Defaults to None.

- Binarize :
  - Number of Otsu Classes : int

    Number of classes into which images are separated.
    Operates by maximizing inter class separation.
    Should correlate to amount of desired materials.
    Must be at least 2 to generate a single threshold value to split image.
    Defaults to 3

  - Number of Classes to Select : int

    Number of classes to select to create binary image from threshold
    calculated threshold values. 1 will select only the pixels with
    intensitites above the uppermost threshold value. Defaults to 1.

- Segment :
  - Min Peak Distance : int

    Minimum distance allowed for distance map peaks or maxima that are used
    to seed watershed segmentation. Can be thought to "grow" into segmented
    regions. Each maxima will correspond to one segmented particle.
    A good way to determine this number is to take the minimum particle
    size (0.075 mm for F50 sand) and divide by spatial resolution (~0.010 mm
    for microfocus x-radiography).
    Defaults to 7

  - Exclude Border Particles : bool

    If True, particles touching the border of the volume (the edges of the
    3D collection of images) will be removed after segmentation.
    Defaults to False

- STL :
  - Create STL Files : bool

    If True, STL files are created post segmentation. Defaults to True

  - Number of Pre-Surface Meshing Erosions : int

    Number of erosions to perform in succession following segmentation of
    particles. Each erosion can be thought of as peeling off the outer "onion
    skin" of particle voxels. Defaults to 0

  - Smooth Voxels with Median Filtering : bool

    If True, smooth particles before marching cubes surface meshing.
    Surface meshing with the marching cubes algorithm produces blocky

  - Marching Cubes Voxel Step Size : int

    Number of voxels to iterate across surface during marching cubes
    algorithm to create surface mesh. Step size 1 creates highest level
    of detail, with larger integers creating coarser meshes.

  - Pixel-to-Length Ratio : float

    Size of a pixel/voxel in the CT images. Used to set scale in STL files.
    Defaults to None.

  - Number of Smoothing Iterations : int or None

    If True, smooth particles following marching cubes surface meshing.
    Surface meshing with the marching cubes algorithm produces blocky
    particles with a limited amount of surface normal directions (simple 6
    Cartesian vectors plus 12 oriented halfway between each of those
    vectors). Defaults to None

  - Target number of Triangles/Faces : int or None

    Desired number of triangles to attempt to reach when simplifying the
    mesh. Mesh will not be simplified if set to None. Defaults to None

  - Simplification factor Per Iteration : int or None

    Factor by which number of triangles will be reduced in each iteration
    while still above target number of triangles. Setting 2 will reduce
    number of triangles. Defaults to None

- Plot :
  - Segmentation Plot Create Figure : bool

    If True, create a segmentation plot that will show routine steps.
    Defaults to False

  - Segmentation Plot Number of Images : int

    If creating segmentation figure, this is the number of images/rows
    in that plot, spaced evenly throughout the slice crop.
    Defaults to 3

  - Segmentation Plot Slices : list of ints or None

    If creating segmentation figure, this can be set to plot specific images
    in the volume. Each integer in the list will correspond to the index
    after slicing. E.g. if segmenting with slice_crop [300, 650], index
    list [0, 50, 100] will plot the images 300, 350, and 400.
    This overrides previous parameter number of image. If None, number of
    images will be used instead. Defaults to None

  - Segmentation Plot Show Maxima : bool

    If creating segmentation plot, this determines whether or not
    maxima/seeds are plotted on the distance map image. Defaults to True

  - Particle Labels Plot Create Figure : bool

    If True, create particle labels figure overlaying particle labels on top
    of slice of segmented volume with unique particle colors.

  - Particle Labels Plot Image Index : int

    If creating particle labels figure, this sets the image index used
    to overlay labels. Uses indices of slice crop; see example in
    segmentation plot slices parameter. Defaults to 0

  - STL Plot Create Figure : False

    If True, creates a figure plotting a random STL file generated in
    this run. Defaults to False

### Preprocessing
Image preprocessing steps include median filter application and intensity
clipping. Applying a median filter to the data reduces noise retaining
edges (unlike Gaussian filtering which will blur edges). Intesity
clipping happens by setting an upper and lower threshold define by
intensity percentile and rescaling the data to that range. Each clipped
intensity is replaced by the value at the bounds of the clip.

### Binarization
Image binarization is performed by applying a multi-Otsu threshold
algorithm to generate threshold values which divide an image into N
regions. This is done by maximizing inter-class variance.

### Segmentation
### Surface Meshing
### Mesh Postprocessing

## [example-ct-to-stl.ipynb](example-ct-to-stl.ipynb)
<a name="example-ct-to-stl.ipynb"></a>
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

## [example-single-particle.ipynb](example-single-particle.ipynb) <a name="example-single-particle.ipynb"></a>
<!------------------------------------------------------------------------>
Workflow example of loading a specific particle from a cropped view of
F50 sand [sample F63](
    https://micromorph.gitlab.io/projectwebsite/ExpDetailsForSample_F63.html),
preprocessing, binarizing, and segmenting the particles within the
cropped region as in the full example. After segmentation, each of the
unique particleID labels are shown overlaid on each particle, a particle
is chosen by selecting its label, then a triangular mesh is created for
that particle only. The individiaul tri-mesh is saved as an STL file
along with its smoothed and simplified versions.

