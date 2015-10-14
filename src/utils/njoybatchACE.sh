#!/bin/bash
#
# NJOY thermr and broadr batch script
# used to doppler broaden resonance peaks.
# Output intended for use with MCNPX
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
#
# notes:
#

# Supply desired output material directory
MATDIR=393K_mat
mkdir $MATDIR

# Supply desired temperature
TEMP='393.'
echo 'NJOY run at' $TEMP 'K'

# Set folder that contains endf data files
# to be processed
FILES=/home/wlg333/njoy/ForMat/MATS/*

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
 20 21
 reconr
 21 22
 'pendf tape from endf/b-vii tape 21'/
 $MAT 3/
 .001/
 'from endf/b tape 20'/
 'processed by the njoy nuclear data processing system'/
 'see original endf/b-vii tape for details of evaluation'/
 0/
 broadr
 21 22 23
 $MAT 1 0 1 0/
 .001/
 $TEMP
 0/
 unresr
 21 23 24
 $MAT 1 7 1
 $TEMP
 1.e10 1.e5 1.e3 100. 10. 1 .1
 0/
 thermr
 0 24 25
 0 $MAT 8 1 1 0 0 1 221 0
 $TEMP
 .001 4.2
 purr
 21 24 25
 $MAT 1 7 20 4/
 $TEMP
 1.e10 1.e5 1.e3 100. 10. 1 .1
 0/
 acer
 21 25 0 26 27/
 1/
 'NJOY CASL'/
 $MAT $TEMP/
 /
 /
 moder
 25 28
 stop
EOF
xnjoy<input

# Write outputs
# use XSn card in mcnpx to load tables.  The xsdirapp file
# created by this script contains the input used on the XSn card(s)
# Ensure to change the "route" strings in the xsdirapp file to 
# the correct path
echo 'saving output, ace, and pendf files'
sed "s/filename/393_$MAT/g" tape27 >> xsdirapp
mv tape26 $MATDIR/393_$MAT
done
