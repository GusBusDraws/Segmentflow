#!/usr/bin/python3

import getopt
import os
import sys
import glob
#from stl import mesh
import stl

from pathlib import Path
import subprocess

# ----------------------------------------
# Utils
# ----------------------------------------

def stripFirstAndLastLines(filename):

    # Read file into list, L, except not the first and last lines
    
    L = []
    
    f = open(Path(filename),'r')
    for line in f:
        L.append(line.replace('\n',''))
    f.close()

    # Read file into list, L
    
    f = open(filename,'w')
    for i in range(0,len(L)):
        if i > 0 and i < len(L)-1:
            print(L[i],file=f)
    f.close()


def clean():
    n_removed = 0
    exts_to_remove = ['.tiff', '.txt', '.stl', '.vtk']
    for file in Path('./').rglob('*'):
        if (
            file.suffix in exts_to_remove
            or file.match('passFailResultsFile')
            or file.match('segment_cake_*.yml')
            or file.match('stdErr*')
            or file.match('tty*')
            or file.match('*~')
        ):
            file.unlink()
    

# ----------------------------------------
# Main: Run case and test for failure
# ----------------------------------------

if __name__ == '__main__':
    
    os.environ["PATH"] += os.pathsep + './'

    clean()

    # -------------------------------------
    # Generate the tiff stack
    # -------------------------------------

    tty = open('tty_genCubeTiffStack','w')
    p = subprocess.run(
        [
            sys.executable, 
            Path('../python/genCubeTiffStack.py'), 
            ('-fgenCubeTiffStack.yml')
        ],
        stdout=tty
    )
    tty.close()

    # -------------------------------------
    # Run the code to be tested: segment.py
    # -------------------------------------
    
    tty      = open('tty','w')
    tty_err = open('stdErr','w')
    p = subprocess.run(
        [sys.executable, Path('../../../segment.py'), ('-fsegment.yml') ],
        stdout=tty, stderr=tty_err
    )
    tty.close()
    tty_err.close()
        
    # -------------------------------------
    # Set up comparison files (STD files)
    # -------------------------------------

    newFiles = ['segmented_002.txt'    ,'segmented_180.txt'    ]
    stdFiles = ['segmented_002.txt_STD','segmented_180.txt_STD']

    # -------------------------------------
    # Convert STL to text
    # -------------------------------------
    
    for i in range(0,len(newFiles)):
    
        # Convert binary stl to ascii

        binarySTL = stl.mesh.Mesh.from_file(newFiles[i].replace('.txt','.stl'))
        binarySTL.save(newFiles[i],mode=stl.Mode.ASCII)
    
        # Strip off the first line, which has a time stamp

        stripFirstAndLastLines(newFiles[i])

    # -------------------------------------
    # Perform comparison
    # -------------------------------------

    option1 = '-f' + str(newFiles).replace('[','').replace(']','').replace("'","")
    option2 = '-s' + str(stdFiles).replace('[','').replace(']','').replace("'","") 
    
    tty      = open('tty_comparator','w')
    tty_err = open('stdErr_comparator','w')

    p = subprocess.run(
        [sys.executable, Path('../../python/comparator_strict.py'),option1,option2],
        stdout=tty, stderr=tty_err
    )

    tty.close()
    tty_err.close()

    # -------------------------------------
    # Make final tty file
    # -------------------------------------

    g = open('tty_runCase','w')
    print('Successful Completion',file=g)
    g.close()
    
    
