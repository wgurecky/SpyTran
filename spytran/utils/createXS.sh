#!/bin/bash

# Generates XS files for Spytran
#
# 1.1) places fissile ENDF files in one direcotry
# 1.2) and non-fissile ENDF files in another directory
# 2) executes NJOY on the ENDF files
# 3) parses NJOY result and write XS files
#

xsdir='XS_default'
fismats='U235 U238 Pu239'
nonfismats='B10 B11 Fe56 H1 Zr90 O16 Cnat'

mkdir MATS
mkdir FISMATS
mkdir $xsdir

for fismat in $fismats
do
    mv endfvii/$fismat FISMATS/.
done
for mat in $nonfismats
do
    mv endfvii/$mat MATS/.
done

### Run njoy
./njoybatchGRPnofis.sh
./njoybatchGRPfis.sh

### Parse njoy output
FILES=./njoyOut/*
for file in $FILES
do
    python2 njoyParse.py -o $file'.xs' -i $file
done

### Move xs files
mv ./njoyOut/*.xs ./$xsdir/.
mv ./$xsdir/C0.xs ./$xsdir/c12.xs
cd ./$xsdir
rename -f 'y/A-Z/a-z/' ./*
