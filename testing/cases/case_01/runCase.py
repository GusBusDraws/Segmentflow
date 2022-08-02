#!/usr/bin/python3

import getopt
import os
import sys
import glob
#from stl import mesh
import stl

from pathlib import Path
import subprocess

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


    

if __name__ == '__main__':
    
    os.environ["PATH"] += os.pathsep + './'

    # Generate the tiff stack

    tty = open('tty','w')
    p = subprocess.run(['python3' , Path('./genCubeTiffStack.py') ,('-fgenCubeTiffStack.yml') ],stdout = tty)
    tty.close()

    # Run test on it
    
    tty      = open('tty','w')
    tty_err = open('stdErr','w')
    p = subprocess.run(['python3' , Path('../../../segment.py'),('-fsegment.yml') ],stdout = tty,stderr=tty_err)
    tty.close()
    tty_err.close()
        
    # Convert binary stl to ascii

    binarySTL = stl.mesh.Mesh.from_file('segmented_02.stl')
    binarySTL.save('segmented_02.txt',mode=stl.Mode.ASCII)
    
    binarySTL = stl.mesh.Mesh.from_file('segmented_63.stl')
    binarySTL.save('segmented_63.txt',mode=stl.Mode.ASCII)
        
    # Strip off the first line, which has a time stamp

    stripFirstAndLastLines('segmented_02.txt')
    stripFirstAndLastLines('segmented_63.txt')
        
        
