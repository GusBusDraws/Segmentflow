#!/usr/bin/python3

#~~~~~~~~~
# Packages
#~~~~~~~~~

import getopt
import os
import sys
import yaml

from pathlib import Path
import subprocess

from utils import *


#~~~~~~~~~~~~~~~~~~~~
# strict comparator
#~~~~~~~~~~~~~~~~~~~~

def comparator_strict(argv):

    newFiles = []
    stdFiles = []
    
    #---------------------------
    # Get command-line arguments
    #---------------------------
    
    try:
        opts, args = getopt.getopt(argv,"hf:s:")
    except getopt.GetoptError:
        fatalError('Error in command-line arguments.  Enter ./segment.py -h for more help')
    yamlFile = ''
    for opt, arg in opts:
        if opt == '-h':
            print("-f [list of new files to compare] -s [list of corresponding STD files]")
            sys.exit()
        if opt == "-f":
            newFiles = arg.replace(' ','').split(',')
        if opt == "-s":
            stdFiles = arg.replace(' ','').split(',')

    #---------------------------
    # Set up defaults
    #---------------------------
    
    failed = '[ FAILED ]'
    passed = '[ passed ]'

    # (1) Read tty file

    try:
        f = open('tty','r')
    except:
        return failed + ' (could not open ./tty file in directory '+os.getcwd()+')'

    L = []
    for line in f:
        L.append(line.replace("\n",""))

    f.close()

    # (2) Make sure the run completed successfully

    badRun = True
    for tmp in L:
       if 'Successful Completion' in tmp:
            badRun = False

    if badRun:
        return failed + ' (segment.py did not complete successfully)'

    # (3) Compare results to the standard file
    #       https://www.quora.com/How-do-I-compare-two-binary-files-in-Python

    filesAreTheSame = True
    
    for i in range(0,len(newFiles)):

        outputFile = newFiles[i]
        stdFile    = stdFiles[i]
        
        filesAreTheSame = open(outputFile, "rb").read() == open(stdFile, "rb").read()
        
        try:
            filesAreTheSame = open(outputFile, "rb").read() == open(stdFile, "rb").read()
        except:
            return failed + ' (error reading output file ' + outputFile + ' or the related _STD file)'

    if not filesAreTheSame:
        return failed + ' (output file ' + outputFile + ' did not match)'

    # (4) Wrap-up, test passes
    
    return passed

    

if __name__ == '__main__':

    os.environ["PATH"] += os.pathsep + './'

    passFailResult = comparator_strict(sys.argv[1:])

    writePassFailResultFile(passFailResult)

    
