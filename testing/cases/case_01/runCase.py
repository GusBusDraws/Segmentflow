#!/usr/bin/python3

import getopt
import os
import sys

if __name__ == '__main__':

    # Generate the tiff stack
    
    os.system('./genCubeTiffStack.py -f genCubeTiffStack.yml > tty_tmp 2> stdErr_tmp')

    # Run test on it
    
    os.system('../../../segment.py -f input.py > tty 2> stdErr')
        
