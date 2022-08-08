#!/usr/bin/python3

import getopt
import os
import sys
import glob
import yaml
#from stl import mesh
import stl
import shutil

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
    p = subprocess.run(
        [
            sys.executable, 
            Path('../python/genCubeTiffStack.py'),
            ('-fgenCubeTiffStack.yml')
        ],
        stdout=tty
    )
    tty.close()

    # Run test on it
    
    tty = open('tty_segmentSupervisor','w')
    tty_err = open('stdErr_segmentSupervisor','w')
    p = subprocess.run(
        [
            sys.executable, 
            Path('../python/segmentSupervisor.py'),
            ('-fsegmentSupervisor.yml')
        ],
        stdout=tty, 
        stderr=tty_err
    )
    tty.close()
    tty_err.close()
        
    # How many pancakes were just now genereated?
    
    stream = open('segmentSupervisor.yml' ,'r')
    yamlDic = yaml.load(stream,Loader=yaml.FullLoader)
    stream.close()

    numCakes = yamlDic['PANCAKES']['Number']

    # Run segment.py on each pancake
    
    for i in range(0,numCakes):
        tty      = open('tty_segment_cake_' + str(i),'w')
        tty_err  = open('stdErr_segment_cake_' + str(i),'w')
        p = subprocess.run(
            [
                sys.executable, 
                Path('../../../segment.py'),
                ('-fsegment_cake_000'+str(i)+'.yml')
            ],
            stdout=tty,
            stderr=tty_err
        )
        tty.close()
        tty_err.close()

    # Eliminate duplicats

    tty = open('tty_elimDups','w')
    tty_err = open('stdErr_elimDupsr','w')
    p = subprocess.run(
        [
            sys.executable, 
            Path('../python/elimDups.py')
        ],
        stdout=tty, 
        stderr=tty_err
    )
    tty.close()
    tty_err.close()
    

    # Since this test is for elimDups.py, first we remove all STL files here
    
    stlList = os.listdir(Path('./'))

    for stlFile in stlList:
        if stlFile.endswith(".stl"):
            os.remove(Path(stlFile))
        
    # Convert binary stl to ascii, stripping off the first and last lines with the time stamps

    testFiles = ['cake_0_13','cake_1_01']

    for t in testFiles:
        stlFile = t + '.stl'
        shutil.copyfile(Path('filteredDups/' + stlFile),Path('./' + stlFile))            # To be consistent with runTests.py, the result needs to be in this directory
        binarySTL = stl.mesh.Mesh.from_file(t+'.stl')
        binarySTL.save(t+'.txt',mode=stl.Mode.ASCII)
        stripFirstAndLastLines(t + '.txt')

    g = open('tty','w')
    print('Successful Completion',file=g)
    g.close()
