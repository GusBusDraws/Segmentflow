#!/usr/bin/python3

#~~~~~~~~~
# Packages
#~~~~~~~~~

import getopt
import os
import sys
import yaml
import shutil

from pathlib import Path
import subprocess


#~~~~~~~~~~
# Utilities
#~~~~~~~~~~

def padString(s,length):

    result = s
    for i in range(0,length - len(s)):
        result += ' '
    return result

def displayListItem(text):
    print(padString(text[0],12) + padString(text[1],20) + '  ' + text[2])

    

    

def fatalError(message):

    print()
    print('==')
    print('||')
    print('||   F A T A L   E R R O R')
    print('||')
    print('||  Sorry, there has been a fatal error. Error message follows this banner.')
    print('||')
    print('==')
    print()
    print(message)
    print()
    exit(0)
    
def help():

    print()
    print('==')
    print('||')
    print('||   This is runTests.py.')
    print('||')
    print('||  This script runs the tests specified in the YAML input file, given')
    print('||  after the -f option.')
    print('||')
    print('||  NOTE: This script must be run from the repository top-level directory.')
    print('||')
    print('==')
    print()
    print('Usage')
    print()
    print('   ./runTests.py -f <inputFile.yml>')
    print()
    print('where <inputFile.yml> is the path to your YAML input file.  See YAML input files')
    print('under ./testing/manage')
    print()
    exit(0)

#~~~~~~~~~~
# Run Case
#~~~~~~~~~~

def runCase(case):

    failed = '[ FAILED ]'
    passed = '[ passed ]'

    # (1) Move to the test case directory
    
    homeDir = os.getcwd()

    try:
        os.chdir( Path('./testing/cases_exe/' + case) )
    except:
        os.chdir(Path(homeDir))
        return failed + ' (could not change into test directory from '+os.getcwd()+')'

    # (2) Run the case

    try:
        p = subprocess.run([sys.executable, Path('runCase.py')])
    except:
        os.chdir(Path(homeDir))
        return failed + ' (could not run ./runCase.py)'

    # (3) Gather results of the test
    
    # (3.1) Read tty file

    try:
        f = open('passFailResultFile','r')
    except:
        os.chdir(Path(homeDir))
        return failed + ' (could not open passFailResultFile for test '+case+')'

    L = []
    for line in f:
        L.append(line.replace("\n",""))
    f.close()

    # (3.2) Return the results
    
    os.chdir(Path(homeDir))
    return L[0]



    
#~~~~~~~~~
# runTests
#~~~~~~~~~

def runTests(argv):

    #---------------------------
    # Copy the cases directory
    #---------------------------

    if os.path.isdir('./testing/cases_exe'):
        shutil.rmtree(Path('./testing/cases_exe'))
        
    shutil.copytree(Path('./testing/cases'),Path('./testing/cases_exe'))
    
    #---------------------------
    # Get command-line arguments
    #---------------------------
    try:
        opts, args = getopt.getopt(argv,"hf:",["ifile=","ofile="])
    except getopt.GetoptError:
        fatalError('Error in command-line arguments.  Enter ./segment.py -h for more help')
    yamlFile = ''
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        if opt == "-f":
            yamlFile = str(arg)

    #---------------------
    # Read YAML input file
    #---------------------
    
    if yamlFile == '':
        fatalError('No input file specified.  Try ./segment.py -h for more help.')
    stream = open(yamlFile, 'r')
    UI = yaml.load(stream,Loader=yaml.FullLoader)   # User Input
    stream.close()

    #---------------------
    # Process user input
    #---------------------

    # (1) This is the list of test cases to be run
    
    listOfTests = UI['Tests']['Cases']

    # (2) Collect descriptions

    descriptions = {}
    for case in listOfTests:
        try:
            descriptions[case] = UI['Descriptions'][case]
        except:
            fatalError("There were no output files specified for case " + case )

    #------------------------
    # Run each test
    #------------------------

    passFail = {}

    for test in listOfTests:
        passFail[test] = runCase(test)
    
    #------------------------
    # Display results
    #------------------------

    for test in passFail:
        displayListItem(  [ passFail[test] , test ,  descriptions[test] ] )




        

if __name__ == '__main__':

    os.environ["PATH"] += os.pathsep + './'
    
    print('')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('RunTests.py')
    print('~~~~~~~~~~~~~~~~~~~~~~~')
    print('')
    runTests(sys.argv[1:])
    print('')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Successful Completion. Bye!')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('')
    print()
        
