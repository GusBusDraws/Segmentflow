#!/usr/bin/python3


import sys, getopt
import glob, os
import numpy
import math
from stl import mesh
import shutil

# ==
# ||
# ||  Help
# ||
# ==

def help():


    os.system('clear')
    print()
    print('-------------------------')
    print('elimDups.py')
    print('-------------------------')
    print()
    print('This Python script loops over all stl files in the **current directory**')
    print('and copies those files to a subdirectory named ./filteredDups, with two')
    print('exceptions: Any unclosed stl files and any stil files that are duplicates ')
    print('of another are not included.  And by "duplicate" we mean that in a non-perfect')
    print('geometrical sense.  We do not expect exact file matches between duplicate shapes')
    print('represented by these stl files.')
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


def distance(a, b):
    return math.sqrt(numpy.sum((a - b) ** 2))


def elimDups(argv):

    # ============================================
    # (1) Process Input (none for now)
    # ============================================

    try:
        opts, args = getopt.getopt(argv,"h",["ifile=","ofile="])

    except getopt.GetoptError:
        print('MakeHTML.py -f <input.db file>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()

    # ============================================
    # (2) Collect list of all stl files
    # ============================================

    listOfFiles = []

    for file in os.listdir("./"):
        if file.endswith(".stl") or file.endswith(".STL"):
            listOfFiles.append(os.path.join("./", file))

    # ============================================
    # (3) Collect values for each stil file
    # ============================================

    centers = []
    volumes = []
    sizes   = []
    badSTLs = []

    for i in range(0,len(listOfFiles)):
        your_mesh = mesh.Mesh.from_file(listOfFiles[i])

        # An error can be thrown here if the STL is unclosed.  We'll
        # use that finding, later, to eliminate this STL.
        
        try:
            volume, cog, inertia = your_mesh.get_mass_properties()
        except:
            volumes.append(-9999.)
            centers.append([0.,0.,0.])
            badSTLs.append(listOfFiles[i])
        
        volumes.append(volume)
        centers.append(cog)

        maxCoord = your_mesh.max_
        minCoord = your_mesh.min_

        sizes.append( [ maxCoord[0] - minCoord[0] ,
                        maxCoord[1] - minCoord[1] ,
                        maxCoord[2] - minCoord[2] ] )

    # ============================================
    # (4) Flag each stl file for copying (or not)
    # ============================================

    # (4.1) Set up the copyIt list:  0 means do not, 1 means do copy it
    #       We default to 1 (copy all files; there are no duplicates or
    #       bad STLs)

    copyIt = []
    for i in range(0,len(listOfFiles)):
        copyIt.append(1)

    # (4.2) Weed out bad STLs (e.g., that are unclosed)

    for i in range(0,len(listOfFiles)):
        if volumes[i] < 0.:
            copyIt[i] = 0

    # (4.3) Weed out duplicates.  Here, loop over all the files

    for i in range(0,len(listOfFiles)):

        # (4.3.1) For each file (listOfFiles[i]), compare to all the
        #         others, looking for a duplicate.

        for j in range(i+1,len(listOfFiles)):

            # (4.3.1a) We need only consider files not yet filtered out.
            #          Also, we do not want to accidentally compare a
            #          file to itself (i.e., i == j)

            if copyIt[j] == 1 and i != j:

                # In each coordinate direction, k, examine the distance 
                # between the two centers of mass and compare that
                # distance to the size of the two particles in that same
                # coordinate direction (k)

                numOverlaps = 0

                for k in range(0,3):
                    distance = abs ( centers[i][k] - centers[j][k] )
                    sizeAverage = ( sizes[i][k] + sizes[j][k] ) / 2.

                    if distance < sizeAverage / 100.: numOverlaps += 1

                # If, for all three coordinate direction, colocation is
                # indicated, we declare the particles to be duplicates:

                if numOverlaps == 3:
                    copyIt[j] = 0

    # ============================================
    # (5) Copy the appropriate files
    # ============================================

    # (5.1) Create the ./filteredDups directory, if needed.  If already present,
    #       clean it out of the old results.

    if not os.path.isdir('./filteredDups'):
        os.system('mkdir filteredDups')
    else:
        os.system('rm filteredDups/*.stl')

    # (5.2) Copy the appropriate files, i.e., all but the duplicates

    for i in range(0,len(listOfFiles)):
        if copyIt[i]:
            shutil.copyfile(listOfFiles[i],'./filteredDups/' + listOfFiles[i])


    # ============================================
    # (6) Log the bad STL files
    # ============================================

    g = open('elimDups.log','w')

    print("",file=g)
    print("Begin elimDups.py log file",file=g)
    print("",file=g)
    print("List of STL files that encountered a center-of-gravity error calculation,",file=g)
    print("probably because they are not closed, geometrically:",file=g)

    for fileName in badSTLs:
        print(fileName,file=g)

    print("",file=g)
    print("End elimDups.py log file",file=g)
    print("",file=g)
    
    g.close()

            

if __name__ == "__main__":

    os.system('clear')

    print()
    print('-------------------------------------')
    print(' Begin Execution' )
    print('-------------------------------------')
    print()

    elimDups(sys.argv[1:])

    print()
    print('-------------------------------------')
    print(' Successful Completion' )
    print('-------------------------------------')
    print()






