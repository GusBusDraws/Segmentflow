#!/usr/bin/python3

import os
from pathlib import Path

    

# ==
# ||
# ||  Help
# ||
# ==

def help():


    os.system('clear')
    print()
    print('-------------------------')
    print('genCubeTiffStack.py')
    print('-------------------------')
    print()
    print('This Python script writes tiff files in the run directory to represent what')
    print('would be produced by a CT scan of a cylinder of binder with grains that')
    print('are shaped like cubes and that are arranged in a well-ordered 3D array.')
    print('This script also produces, in the "paraview" directory under the current')
    print('run directory, Paraview output files for visualization.')
    print('')
    print('To view the files in Paraview,  open the vtk file and click "Apply".  Then,')
    print('with the file highlighted in the GUI, click the "Threshold" option, which ')
    print('will create a new plot object under the file.  Adjust the "Threshold" value')
    print('in Paraview to reveal the shapes in the vtk file.')
    print()



# ==
# ||
# ||  fatalError
# ||
# ==

def fatalError(message):

    print()
    print('-------------------------')
    print('Fatal Error')
    print('-------------------------')
    print()
    print('Error Message:')
    print()
    print(message)
    print()

    exit(0)

# ==
# ||
# ||  fatalError
# ||
# ==

def padInteger(iVal):

    if iVal < 10:    return '000' + str(iVal)
    if iVal < 100:   return  '00' + str(iVal)
    if iVal < 1000:  return   '0' + str(iVal)
    if iVal < 10000: return         str(iVal)

    fatalError('padInteger:  iVal too big')



    
def clear_directory(directory):
    for f in os.listdir(Path(directory)):
        if '.stl' in f:
            os.remove(Path(directory + "/" + f))
    
