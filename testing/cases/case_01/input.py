Files :

   CT Scan Dir : './'
   STL Dir :  './'
   STL Prefix : 'segmented_'
   Overwrite Existing STL Files : True
   Particle ID : 
   Suppress Save Messages : False

Load :

   File Suffix : 'tiff' 
   Slice Crop :
   Row Crop :
   Col Crop :

Preprocess:

   Apply Median Filter : True
   Rescale Intensity Range : [5, 95]

Binarize :

   Number of Otsu Classes : 2
   Number of Classes to Select : 1

Segment :

   Use Integer Distance Map : True
   Min Peak Distance : 2
   Exclude Border Particles : True

STL :

   Erode Particles : False
   Marching Cubes Voxel Step Size : 1
   Pixel-to-Length Ratio : 1.0

Plot :

   Show Segmentation Figure : False
   Number of Images : 3
   Plot Maxima : True
   Show Particle Labels Figure : False
   Particle Label Image Index : 50
   Show Random STL Figure : False





