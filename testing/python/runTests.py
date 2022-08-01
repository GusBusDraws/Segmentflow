#!/usr/bin/python3

#~~~~~~~~~
# Packages
#~~~~~~~~~

import getopt
import os
import sys
import yaml

#~~~~~~~~~~
# Utilities
#~~~~~~~~~~

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

def runCase(case,outputFiles):

    failed = '[ FAILED ]'
    passed = '[ passed ]'

    # (1) Move to the test case directory
    
    homeDir = os.getcwd()

    try:
        os.chdir('./testing/cases/' + case )
    except:
        return failed + ' (could not change into test directory)'

    # (2) Clean out old results

    for outputFile in outputFiles:
        if os.path.isfile(outputFile):
            os.remove(outputFile)
            
    if os.path.isfile('tty'):
        os.remove('tty')
    
    # (3) Run the case

    try:
        os.system('./runCase.py')
    except:
        return failed + ' (could not run ./runCase.py)'

    # (4) Compare results to "standard" file

    # (4.1) Read tty file

    try:
        f = open('tty','r')
    except:
        return failed + ' (could not open ./tty file)'

    L = []
    for line in f:
        L.append(line.replace("\n",""))

    f.close()

    # (4.2) Make sure the run completed successfully

    badRun = True
    for tmp in L:
        if 'Successful Completion' in tmp:
            badRun = False

    if badRun:
        return failed + ' (segment.py did not complete successfully)'

    # (4.3) Compare results to the standard file
    #       https://www.quora.com/How-do-I-compare-two-binary-files-in-Python

    filesAreTheSame = True
    
    for outputFile in outputFiles[case]:
        print('outputfile = ',outputFile)

        stdFile = outputFile + "_STD"
        
        try:
            filesAreTheSame = open(outputFile, "rb").read() == open(stdFile, "rb").read()
        except:
            return failed + ' (error reading output file ' + outputFile + ' or the related _STD file)'

    if not filesAreTheSame:
        return failed + ' (output files did not match)'

    # (5) Wrap-up
    
    os.chdir(homeDir)

    # (6) Test passes

    return passed

    
#~~~~~~~~~
# Workflow
#~~~~~~~~~

def runTests(argv):
    
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

    # (2) For each test case, there is at least one "std" file that is to be compared against

    outputFiles = {}
    for case in listOfTests:
        try:
            outputFiles[case] = UI['Output Files'][case]
        except:
            fatalError("There were no output files specified for case " + case )

    #------------------------
    # Run each test
    #------------------------

    passFail = {}

    for test in listOfTests:
        passFail[test] = runCase(test,outputFiles)
    
    #------------------------
    # Display results
    #------------------------

    for test in passFail:
        print(test + "  ---> " + passFail[test])


if __name__ == '__main__':
    os.system('clear')
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
        
