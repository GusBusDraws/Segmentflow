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
    

def writePassFailResultFile(result):
    g = open(Path('passFailResultFile'),'w')
    print(result,file=g)
    g.close()

    print(result)
    


    

