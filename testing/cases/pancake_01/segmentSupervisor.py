#!/usr/bin/python3


import sys, getopt
import glob, os
import numpy
import math
import yaml
from stl import mesh
import shutil
from pathlib import Path
import subprocess

# ==
# ||
# ||  Help
# ||
# ==

def help():


    os.system('clear')
    print()
    print('-------------------------')
    print('segmentSupervisor.py')
    print('-------------------------')
    print()
    print('This Python script supervises the execution of a subsidiary script called')
    print('segment.py.  This script runs segment.py on "chunks" of the CT scan because')
    print('for large CT scans, too much memory is required to process it all at once.')
    print('This script divides up the task based on tiff file ID numbers, which means')
    print('that for cylindrical samples, the chunks together look like a stack of')
    print('pancakes.')
    print()
    print('-------------------------')
    print('User Inputs')
    print('-------------------------')
    print()
    print('Inputs are provided through a YAML file specified by a command-line argument, i.e., ')
    print('./segmentSupervisor -f myYamlInput.yml.   The input file is described below.')
    print('')
    print()
    print('-------------------------')
    print('YAML Input File Structure')
    print('-------------------------')
    print()
    print('FILES:')
    print('   segment.py : "path to segment.py, in quotes"       ')
    print('   segment.py Input : "path to the segment.py input file (used as a template), in quotes" ')
    print('')
    print('PANCAKES:')
    print('   Number : Integer number of chunks       ')
    print('   Overlap : Integer number of tiff files to overlap between chunks       ')
    print('')
    print('-------------------------')
    print('Your Workflow')
    print('-------------------------')
    print('')
    print('(1) Create a yml input file for this script. per the specifications above.')
    print('(2) Check your yml template input file for segment.py is exactly how you')
    print('    want it. Pay special attention to its file location specification as well as ')
    print('    the tiff file range. Note that the STL Prefix in the template file will be')
    print('    appended with cake_<Index>_')
    print('(3) Run this script.')
    print('(4) Take a look at ./makeAllPancakes, and one of the input files it uses.')
    print('(5) Run ./makeAllPancakes')
    print('(6) Then run elimDups.py, which is in the same repo and directory as this python script.')
    print('')
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

def padInteger(iVal):

    if iVal < 10:    return '000' + str(iVal)
    if iVal < 100:   return  '00' + str(iVal)
    if iVal < 1000:  return   '0' + str(iVal)
    if iVal < 10000: return         str(iVal)
    
    fatalError('padInteger:  iVal too big')

# ==
# ||
# || writeYamlAndExecuteSegment
# ||
# ==

def writeYamlAndExecuteSegment(yamlDic_exec, templateFile , idx, segmentPy, segmentBashScriptFile):

    # -------------------------------------------------------------------------
    # (1) Write a a workable yaml file for segment.py based on templateFile
    # -------------------------------------------------------------------------

    if ".yml" not in templateFile:
            fatalError("semgemt.py's template input file must have a .yml extension.  Sorry...")

    thisYamlFile = templateFile.replace('.yml','_cake_' + padInteger(idx) + '.yml')
    g = open(thisYamlFile,'w')

    # This line appends the segmentBashScriptFile with the segment.py execution command for this pancake

    print(segmentPy + " -f " + thisYamlFile,file=segmentBashScriptFile)
    

    for category in yamlDic_exec:
        print('',file=g)
        print(category + ":"  ,file=g)
        print('',file=g)

        for subCat01 in yamlDic_exec[category]:
            value = str(yamlDic_exec[category][subCat01])

            addQuotes = ''
            if 'Suffix' in subCat01: addQuotes = "'"
            if 'Prefix' in subCat01: addQuotes = "'"
            if 'Dir'    in subCat01: addQuotes = "'"

            try:
                value = value.replace('None','')
            except:
                pass
            
            print("   " + subCat01 + " : "  + addQuotes + str(value) + addQuotes ,file=g)
        
    g.close

    # -------------------------------------------------------------------------
    # (2) Execute segment.py on that workable yaml file
    # -------------------------------------------------------------------------
    
#    tty      = open('tty_'+thisYamlFile.replace('.yml',''),'w')
#    tty_err = open('stdErr_'+thisYamlFile.replace('.yml',''),'w')
#    print("Running " + thisYamlFile)
#    p = subprocess.run(['python3' , Path('../../../segment.py'),('-f'+thisYamlFile) ],stdout = tty,stderr=tty_err)
#    tty.close()
#    tty_err.close()


    
# ==
# ||
# ||
# || s e g m e n t S u p e r v i s o r
# ||
# || Main routine for this file
# ||
# ==    

def segmentSupervisor(argv):


    inputFile = ''

    # ============================================
    # (1) Process Input (none for now)
    # ============================================

    try:
        opts, args = getopt.getopt(argv,"hf:",["ifile=","ofile="])

    except getopt.GetoptError:
        print('MakeHTML.py -f <input.db file>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        if opt == '-f':
            inputFile = arg

    if inputFile == '': fatalError("Specify your yaml input file using -f.  Use -h for more help")


    # ============================================
    # (2) Process User Input
    # ============================================

    stream = open(inputFile, 'r')
    yamlDic = yaml.load(stream,Loader=yaml.FullLoader)
    stream.close()

    print('yamlDic = ',yamlDic)
    
    try:
        segmentPy = yamlDic['FILES']['segment.py']
    except:
        fatalError("Path to segment.py not specified")

    try:
        templateFile = yamlDic['FILES']['segment.py Input']
    except:
        fatalError("segment.py template input file not specified.")

    try:
        numCakes = yamlDic['PANCAKES']['Number']
    except:
        fatalError("Number of chunks (pancakes) not specified.")

    try:
        tiffOverlap = yamlDic['PANCAKES']['Overlap']
    except:
        fatalError("Number of tiff files to overlap between pancakes not specified.")

        
    # ============================================
    # (3) Process the segment.py template
    # ============================================

    stream = open(templateFile, 'r')
    yamlDic_template = yaml.load(stream,Loader=yaml.FullLoader)
    stream.close()

    try:
        tiffRange = yamlDic_template['Load']['Slice Crop']
    except:
        fatalError("In your segment.py template script, you must specify Slice Crop for this script to function.")
    
    # ============================================
    # (4) Compute pancakes
    # ============================================
    
    cakeRanges = []

    tiffsPerCake = int ( ( tiffRange[1] - tiffRange[0] ) / numCakes )

    print("(o) Tiff range to be processed in chunks, per segment.py template file: ",tiffRange)
    print("(o) Number of cakes, per user input: ",numCakes)
    print("(o) tiffsPerCake = ",tiffsPerCake)

    # ============================================
    # (4) Execute segment.py for each pancake
    # ============================================

    yamlDic_exec = yamlDic_template

    tiff0 = tiffRange[0]
    tiff1 = 0

    idx = 0

    segmentBashScriptFile = open('makeAllPancakes','w')
    
    while tiff1 != tiffRange[1]:
        
        # Compute the tiff range for this pancake
        
        tiff1 = tiff0 + tiffsPerCake - tiffOverlap    # Upper tiff limit decreased for overlap factor
        tiff1 = min(tiff1,tiffRange[1])               # Don't go beyond the original range
        
        # Modify yaml dictionary for this pancake
        
        yamlDic_exec['Files']['STL Prefix'] = 'cake_' + str(idx) + '_'
        yamlDic_exec['Load']['Slice Crop']  = [tiff0,tiff1]
        
        # Write input file and add a line to the bash script that the user will ultimately execute

        writeYamlAndExecuteSegment(yamlDic_exec,templateFile,idx,segmentPy,segmentBashScriptFile)

        # Get set up for the next pancake

        tiff0 = tiff1 - tiffOverlap

        if tiff0 < 0: fatalError("Something is wrong in the pancake range computation.  tiff0 is negative.")

        idx += 1


    segmentBashScriptFile.close()



        
if __name__ == "__main__":

    os.environ["PATH"] += os.pathsep + './'

    print()
    print('-------------------------------------')
    print(' Begin Execution: segmentSupervisor' )
    print('-------------------------------------')
    print()

    segmentSupervisor(sys.argv[1:])

    print()
    print('-------------------------------------')
    print(' Successful Completion' )
    print('-------------------------------------')
    print()






