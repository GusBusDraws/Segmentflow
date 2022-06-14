# Segmentflow
Developed by C. Gus Becker (GitHub/GitLab: @cgusb).

This project establishes a segmentation workflow for 3D image data obtained from processes like x-ray computed tomography. Segmented data can be expressed as integer-labeled images (integer value of pixels correspond to unique particles) or separate STL files.

## Contents

### [segment.py](segment.py)
Module containing segmentation workflow functions.

### [ct-to-stl.ipynb](ct-to-stl.ipynb)
General workflow example for entire segmentation process. Load a cropped region of an F50 sand sample (PetaLibrary/CT Scans/Sand Compression in Steps/NoComptiff/), segment each particle, convert each particle to a triangular mesh, and save each tri-mesh as a separate STL file.

### [single-particle.ipynb](single-particle.ipynb)
Workflow example of loading a cropped region of the same F50 sand sample (PetaLibrary/CT Scans/Sand Compression in Steps/NoComptiff/), segmenting each particle, showing the labels of each particle, picking a particle by its label, creating a triangular mesh for the particle using a marching cubes algorithm, and saving the individiaul tri-mesh as an STL file.

