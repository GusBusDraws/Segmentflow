#!/usr/bin/python3

import getopt
import os
import sys
import glob

from pathlib import Path
import subprocess

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
        
        
