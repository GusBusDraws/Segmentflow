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

def radiusSquaredOfPoint(Grid,i,j):
    
    # Compute the center of the circle, which is centered in the middle of the grid, i.e., at half the "domain"  value
            
    xc =  Grid['domain'][0] / 2.
    yc =  Grid['domain'][1] / 2.
    
    # Compute the distance (in the z-plane) from this i-j grid point from the circle's center
    
    dx = i*Grid['delta'][0] - xc
    dy = j*Grid['delta'][1] - yc
    rSquared = dx*dx + dy*dy

    return rSquared




def isInsideTheSample(Grid,i,j):

    rCircle = Grid['circle radius']
    
    if radiusSquaredOfPoint(Grid,i,j) < rCircle*rCircle : return True

    return False

def isInsideTheInnerCircle(Grid,i,j):

    rCircle = Grid['inner circle radius']
    
    if radiusSquaredOfPoint(Grid,i,j) < rCircle*rCircle : return True

    return False




def xyInsideCircle(Grid,x,y):

    # Compute the center of the circle, which is centered in the middle of the grid, i.e., at half the "domain"  value
            
    xc =  Grid['domain'][0] / 2.
    yc =  Grid['domain'][1] / 2.

    try:
        rCircle = Grid['inner circle radius']
    except:
        fatalError('You must set the "inner circle radius" value.')
    
    # Compute the distance (in the z-plane) from this i-j grid point from the circle's center
    
    dx = x - xc
    dy = y - yc
    rSquared = dx*dx + dy*dy
    
    if rSquared < rCircle*rCircle : return True

    return False



# ==
# || 
# || Sets brightness levels for a 3D grid points (grid3D) in Grid dictionary
# || inside hexahedrals defined by shapeDic
# || 
# ==

def setBrightnessForHexes(Grid,shapeDic,brightness,grid3D):
    x = 0.
    y = 0.
    z = 0.
    k = 0
    corner1 = [0.,0.,0.]
    corner2 = [0.,0.,0.]
    corner3 = [0.,0.,0.]
    corner4 = [0.,0.,0.]
    
    for i in range(0,shapeDic['num'][0]):
        for j in range(0,shapeDic['num'][1]):
            for k in range(0,shapeDic['num'][2]):

                # Compute the bottom, lower-left corner of the
                # cube and the top, upper-right corner of the cube
            
                for n in range(0,3):

                    corner1[0] = shapeDic['offset'][0] + i * ( shapeDic['size'][0] +  shapeDic['spacing'][0] )
                    corner1[1] = shapeDic['offset'][1] + j * ( shapeDic['size'][1] +  shapeDic['spacing'][1] )
                    corner1[2] = shapeDic['offset'][2] + k * ( shapeDic['size'][2] +  shapeDic['spacing'][2] )
                        
                    for kk in range(0,3):
                        corner3[kk] = corner1[kk]  # Will adjust in a moment
                        corner4[kk] = corner1[kk]  # Will adjust in a moment
                        corner2[kk] = corner1[kk] + shapeDic['size'][kk]

                    # Other two corners

                    corner3[0] = corner1[0] + shapeDic['size'][0]   # x direction only
                    corner4[1] = corner1[1] + shapeDic['size'][1]   # y direction only

                    # Check to ensure the entire cube is inside the inner circle

                    ptclInside = False
                    if xyInsideCircle(Grid,corner1[0],corner1[1]):
                        if xyInsideCircle(Grid,corner2[0],corner2[1]):
                            if xyInsideCircle(Grid,corner3[0],corner3[1]):
                                if xyInsideCircle(Grid,corner4[0],corner4[1]):
                                    ptclInside = True
                                

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
                            if isInsideTheInnerCircle(Grid,ii,jj) and ptclInside :            
                                for kk in range(idxStart[2],idxEnd[2]):
                                    try:
                                        grid3D[ii][jj][kk] = brightness
                                    except:
                                        pass


    return grid3D


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

    try:
        Voids  = UI['Voids']
    except:
        Voids = []

    try:
        Ptcls  = UI['Particles']
    except:
        Ptcls = []

    try:
        Grid   = UI['Grid']
    except:
        fatalError('Grid: input is required in the yaml input file.')

    try:
        Binder = UI['Binder']
    except:
        Binder = []

    try:
        tiffDir = UI['Files']['tiff save dir']
    except:
        print('(o) tiff files will be written to the current directory by defult.  Change this by setting "tiff save dir" under "Files" in the yml file.')
        tiffDir = './'

    print('(o) tiff dir: ' + tiffDir)
    
    if not os.path.isdir(Path(tiffDir)):
        os.mkdir(Path(tiffDir))

    # ============================================
    # Process User Input
    # ============================================
    #

    try:
        binderBrightness = Grid['brightness']
    except:
        binderBrightness = 100.

    try:
        ptclBrightness = Ptcls['brightness']
    except:
        ptclBrightness = 65535.

    try:
        binderBrightness = Grid['brightness']
    except:
        binderBrightness = 100.

    try:
        voidBrightness = Voids['brightness']
    except:
        voidBrightness = 0.

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

    if len(Ptcls) > 0:
        grid3D = setBrightnessForHexes(Grid,Ptcls,ptclBrightness,grid3D)

    if len(Voids) > 0:
        grid3D = setBrightnessForHexes(Grid,Voids,voidBrightness,grid3D)
    

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

    clear_directory(tiffDir)

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

        rawtiff.save(Path(tiffDir + '/' + fileName))
                
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






