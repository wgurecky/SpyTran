#!/bin/bash
#
# Njoy groupr batch run script
# Output intended for use with multigrp transport codes
#
# Author:
# William Gurecky
# william.gurecky@utexas.edu
#
# Change Log:
# ------------
# 10/14/2013  Creation, issues with NJOY99 reading endfVii files
# 2/4/2014    NJOY99 Updated to 99.396 (thanks Matt Montgomery)
#             allowing NJOY to process some endfVii files
# 2/5/2014    Correct resulting xsdir file so filepath doesnt
#             need manual modification
# 2/15/2014   Place material data in designated folder
# 2/4/2014    NJOY99 Updated to NJOY2012 (thanks Chris Van Der Hooeven)
#             allowing NJOY to process most endfVii files
# 10/12/2015  Moved from ACE to Multigrp output
#
# notes:
#

# Supply desired OUTPUT material directory
MATDIR=ForCompMethods
mkdir $MATDIR

# Supply desired temperature
TEMP='300.'
GRPSTR='1.E-3 1.E-2 1.E-1 1. 1.E1 1.E2 1.E3 1.E4 1.E5 1.E6 1.E7/'
NGN='10'
SIGZ='1.E10 1.E5 1.E3 1.E2 1.E1 1.E0 1.E-1/'
NSIGZ='7'
LORD='8'

echo 'NJOY run at' $TEMP 'K'
# Set folder that contains endf data files
# to be processed
FILES=/home/wlg333/school/njoy/FISMATS/*

# Run NJOY
for file in $FILES
do
MAT=`awk '/VII[ ]+MATERIAL/ {print $3}' $file `
echo 'Material: ' $MAT
echo 'Getting endf tape...'
cp $file tape20
echo 'running njoy'
cat>input <<EOF
 moder
 20 -21
 reconr
 -21 -22
 'pendf tape for $MAT'/
 $MAT 3/
 .005/
 '$MAT from endf/b-vi'/
 'processed by the njoy nuclear data processing system'/
 'see original endf/b-vi tape for details of evaluation'/
 0/
 broadr
 -21 -22 -23
 $MAT 1 0 1 0./
 .005/
 $TEMP
 0/
 unresr
 -21 -23 -24
 $MAT 1 $NSIGZ 1/
 $TEMP
 $SIGZ
 0/
 groupr
 20 -24 0 -25/
 $MAT 1 0 4 $LORD 1 $NSIGZ/
 '$MAT fission xs and scattering kernel'/
 $TEMP
 $SIGZ
 $NGN /
 $GRPSTR
 1.0 0.0235 100000. 1000000./
 3 1/
 3 18/
 6 2/
 3 452/
 6 18/
 0/
 0/
 moder
 -25 28
 stop
EOF
xnjoy<input

# Write outputs
echo 'saving output'
mv output $MATDIR/njoyout_$MAT
done
#
#
#
