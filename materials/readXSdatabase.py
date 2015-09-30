# Reads all *.xs files found in the desired directory
# Populates material classes
#
import os
import readXS


def readXSFolder(folderPath):
    xsTables = {}
    for (dirpath, dirnames, filenames) in os.walk(folderPath):
        for filename in filenames:
            if filename.endswith('.xs'):
                print("loading xs file: " + os.path.join(dirpath, filename) + "  ..."),
                try:
                    xsTables[os.path.splitext(filename)[0]] = \
                        readXS.readXS(os.path.join(dirpath, filename))
                    print("sucess!")
                except:
                    print("failure.")
            else:
                pass
    print("Cross section database sucessfully loaded.")
    return xsTables


if __name__ == "__main__":
    dataDict = readXSFolder('./hw2')
