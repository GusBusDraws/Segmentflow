#!/usr/bin/python3


# ==
# ||
# ||  genCubeTiffStack.py
# ||
# ||  Script for writing a tiff stack to test segmentation software and end-to-end
# ||  micromorphic calculations
# ||
# ||  Scott Runnels
# ||
# ==

import sys, getopt
import os
import numpy
import yaml
from PIL import Image
from paraviewWriter import *
from utils import *

# ==
# || 
# || isInsideTheSample: Given an i-j point location associated with the "Grid", return true
# || if the corresponding physical point is inside the circle.
# || 
# ==

def isInsideTheSample(Grid,i,j):
    
    # Compute the center of the circle, which is centered in the middle of the grid, i.e., at half the "domain"  value
            
    xc =  Grid['domain'][0] / 2.
    yc =  Grid['domain'][1] / 2.
    
    # Compute the distance (in the z-plane) from this i-j grid point from the circle's center
    
    dx = i*Grid['delta'][0] - xc
    dy = j*Grid['delta'][1] - yc
    rSquared = dx*dx + dy*dy

    # Test if inside the circle

    rCircle = Grid['circle radius']

    if rSquared < rCircle*rCircle: return True

    return False




# ==
# || 
# || Main Routine
# || 
# ==

def main(argv):

    # ============================================
    # Default Values
    # ============================================

    yamlFile = ''
    ptclBrightness = 65535.
    binderBrightness = 100.

    # ============================================
    # Process Command-Line Options
    # ============================================

    try:
        opts, args = getopt.getopt(argv,"f:",["ifile=","ofile="])

    except getopt.GetoptError:
        print("Error in input parameters.  Run with -h to get help.")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        if opt == "-f":
            yamlFile = arg

    if yamlFile == '': fatalError("You must specify a yaml input file using -f")
    
    stream = open(yamlFile, 'r')
    UI = yaml.load(stream,Loader=yaml.FullLoader)   # User Input
    stream.close()

    Ptcls = UI['Particles']
    Grid  = UI['Grid']

    # ============================================
    # Check Input
    # ============================================

    for kk in range(0,3):

        if Ptcls['num'][kk] * ( Ptcls['size'][kk] + Ptcls['spacing'][kk] ) > Grid['domain'][kk]:
            fatalError('In the ' + str(kk) + ' direction, the domain size does not accommodate all of the particles.')

    # ============================================
    # Compute geometry
    # ============================================

    Grid['delta'] = [0.,0.,0.]
    for k in range(0,3):
        Grid['delta'][k] = Grid['domain'][k]/Grid['num'][k]

    # ============================================
    # Create 3D array for storing cubes
    # ============================================

    # grid3D stores the brightness level in a 3D grid of pixels, i.e. voxels.  We start by
    # filling it with zeros.
    
    grid3D  = numpy.zeros( ( Grid['num'][0] ,  Grid['num'][1] ,  Grid['num'][2]  ) )

    # Give a low but non-zero brightness level to the voxels inside the sample, i.e., inside the circle (at all z levels)

    for i in range(0,Grid['num'][0]):
        for j in range(0,Grid['num'][1]):
            if isInsideTheSample(Grid,i,j):            
                for k in range(0,Grid['num'][2]):
                    grid3D[i][j][k] = binderBrightness
        
    # ============================================
    # Populate the 3D array
    # ============================================

    x = 0.
    y = 0.
    z = 0.
    k = 0
    corner1 = [0.,0.,0.]
    corner2 = [0.,0.,0.]
    for i in range(0,Ptcls['num'][0]):
        for j in range(0,Ptcls['num'][1]):
            for k in range(0,Ptcls['num'][2]):

                # Compute the bottom, lower-left corner of the
                # cube and the top, upper-right corner of the cube
            
                for n in range(0,3):

                    corner1[0] = Ptcls['offset'][0] + i * ( Ptcls['size'][0] +  Ptcls['spacing'][0] )
                    corner1[1] = Ptcls['offset'][1] + j * ( Ptcls['size'][1] +  Ptcls['spacing'][1] )
                    corner1[2] = Ptcls['offset'][2] + k * ( Ptcls['size'][2] +  Ptcls['spacing'][2] )
                        
                    for kk in range(0,3):
                        corner2[kk] = corner1[kk] + Ptcls['size'][kk]

                    # Find the beginning and ending grid indices in the x, y, and z directions
                    # that are inside this cube

                    idxStart = [0,0,0]
                    idxEnd   = [0,0,0]

                    for n in range(0,3):
                        idxStart[n] = int(corner1[n] / Grid['delta'][n])
                        idxEnd[n]   = int(corner2[n] / Grid['delta'][n])

                    # Loop over those grid points, setting the brightness level to 1, thereby
                    # indicating they are in a particle
                    
                    for ii in range(idxStart[0],idxEnd[0]):
                        for jj in range(idxStart[1],idxEnd[1]):
                            if isInsideTheSample(Grid,ii,jj):            
                                for kk in range(idxStart[2],idxEnd[2]):
                                    try:
                                        grid3D[ii][jj][kk] = ptclBrightness
                                    except:
                                        out = "Bounds error \n"
                                        out += "Size of grid3d: " + str(Grid['num']) + "\n"
                                        out += "Deltas in grid3d: " + str(Grid['delta']) + "\n"
                                        out += "idxStart: " + str(idxStart) + "\n"
                                        out += "idxEnd: " + str(idxEnd) + "\n"
                                        out += "corner1: " + str(corner1) + "\n"
                                        out += "corner2: " + str(corner2) + "\n"
                                        out += "ii,jj,kk: " + str(ii) + ' ' + str(jj) + ' ' + str(kk)
                                        fatalError (out)

    # ============================================
    # Write the normalized araview File
    # ============================================

    writeParaview('cubes-written_to_tiffs.vtk',grid3D,Grid['num'][0],Grid['num'][1],Grid['num'][2])
    
    # ============================================
    # Write the tiff stack
    # ============================================

    # Initialize a 2D grid for use in writing each z-level of grid3d
    
    imarray = numpy.zeros( ( Grid['num'][0] ,  Grid['num'][1]                    ) ).astype(numpy.uint16)
    print("size of  imarray: ",imarray.shape)
    
    # Set up tiff image format

    # Loop over z-direction of grid3d, writing a tiff file for each z-level

    for k in range(0,Grid['num'][2]):
        
        # Store this z-level in 2D array
        
        for i in range(0,Grid['num'][0]):
            for j in range(0,Grid['num'][1]):
                imarray[i][j] = grid3D[i][j][k]

        # Create PIL Image object from the 2D array

        rawtiff = Image.fromarray(imarray)

        # tiff filename

        fileName = 'cubes_' + padInteger(k) + '.tiff'

        # Write the tiff file

        rawtiff.save(fileName)
                
    return




if __name__ == "__main__":

    print()
    print('==')
    print('||')
    print('||')
    print('||  g e n C u b e T i f f S t a c k')
    print('||')
    print('||')
    print('==')
    print()
    print('-------------------------------------')
    print(' Begin Execution' )
    print('-------------------------------------')
    print()

    main(sys.argv[1:])

    print()
    print('-------------------------------------')
    print(' Successful Completion' )
    print('-------------------------------------')
    print()





