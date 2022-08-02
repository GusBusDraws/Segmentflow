#!/usr/bin/python3

import sys, getopt
import os


# ==
# ||
# ||  Write 3D Paraview output
# ||
# ==

def writeParaview(fileName,xyzData,nX,nY,nZ):

    print()
    print('--------------------------------------------------------------------')
    print('(o) Writing the Paraview file ' + fileName)
    print('--------------------------------------------------------------------')
    print()

    # Open file
    
    g = open(fileName,'w')

    # Write metadata
    
    print('# vtk DataFile Version 2.0',file=g)
    print('Really cool data',file=g)
    print('ASCII',file=g)
    print('DATASET STRUCTURED_POINTS',file=g)
    print('DIMENSIONS ' , nX , ' ' , nY , ' ' , nZ,file=g)
    print('ASPECT_RATIO 1 1 1 ',file=g)
    print('ORIGIN 0 0 0 ',file=g)
    print(' ',file=g)

    numPts = nX*nY*nZ
    print('POINT_DATA ', numPts,file=g)
    print('SCALARS Brightness float',file=g)
    print('LOOKUP_TABLE default',file=g)

    # # Write field data

    print()
    print("    (+) Printing "+str(nZ)+" levels: ",end="")
    for k in range(0,nZ):
        print(" " , k,end="" )
        for j in range(0,nY):
            for i in range(0,nX):
                print(xyzData[i][j][k],file=g)

    print()
    print()
    
    g.close()


    
