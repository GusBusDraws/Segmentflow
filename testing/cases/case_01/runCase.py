#!/usr/bin/python3

import getopt
import os
import sys

if __name__ == '__main__':

    # Generate the tiff stack
    
    os.system('./genCubeTiffStack.py -f genCubeTiffStack.yml > tty_tmp 2> stdErr_tmp')

    # Run test on it
    
    os.system('../../../segment.py -f segment.yml > tty 2> stdErr')
        
    # Collect all the stl files
    
    os.system('ls *.stl > listOfSTLfiles')
        
