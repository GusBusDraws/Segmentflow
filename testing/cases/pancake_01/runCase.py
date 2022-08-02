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

    tty = open('tty_genCubeTiffSTack','w')
    p = subprocess.run(['python3' , Path('../case_01/genCubeTiffStack.py') ,('-fgenCubeTiffStack.yml') ],stdout = tty)
    tty.close()

    # Run test on it
    
    tty      = open('tty_segmentSupervisor','w')
    tty_err = open('stdErr_segmentSupervisor','w')
    p = subprocess.run(['python3' , Path('./segmentSupervisor.py'),('-fsegmentSupervisor.yml') ],stdout = tty,stderr=tty_err)
    tty.close()
    tty_err.close()
        
    # Run segment.py on each pancake

    for i in range(0,5):
        tty      = open('tty_segment_cake_' + str(i),'w')
        tty_err  = open('stdErr_segment_cake_' + str(i),'w')
        p = subprocess.run(['python3' , Path('../../../segment.py'),('-fsegment_cake_000'+str(i)+'.yml') ],stdout = tty,stderr=tty_err)
        tty.close()
        tty_err.close()
    
    # Convert binary stl to ascii

    binarySTL = stl.mesh.Mesh.from_file('cake_0_001.stl')
    binarySTL.save('cake_0_001.txt',mode=stl.Mode.ASCII)
    
    binarySTL = stl.mesh.Mesh.from_file('cake_0_010.stl')
    binarySTL.save('cake_0_010.txt',mode=stl.Mode.ASCII)

        
    # Strip off the first line, which has a time stamp

    stripFirstAndLastLines('cake_0_001.txt')
    stripFirstAndLastLines('cake_0_010.txt')
        
    # Make final tty file

    g = open('tty','w')
    print('Successful Completion',file=g)
    g.close()
