#!/usr/bin/python
import os


matLib = {}
try:
    raise
    import isoDictGen as isoGen
    matLib = isoGen.genIsoDict()
except:
    print("Reading isotopic information.")
    import json
    with open(os.path.dirname(os.path.relpath(__file__)) + '/isoDict.txt', 'r') as infile:
        matLib = json.load(infile)
