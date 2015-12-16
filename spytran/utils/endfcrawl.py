#!/usr/bin/python
#
# Crawls the T2-Data website for endf data libraries
# places endf files in designated output directory
#
# 12/4/2013
# William Gurecky
#

import mechanize as mz
import re


#outdir = 'endfvii/'
outdir = 'endfvi/'

# Set target page
#target = 'http://t2.lanl.gov/nis/data/endf/endfvii-n.html'
target = 'http://t2.lanl.gov/nis/data/endf/endfvi-n.html'

# Open up browser instance
br = mz.Browser()
br.open(target)

keywrd = re.compile("neutron")
links = list(br.links())
for link in links:
    if keywrd.findall(link.url):
        print("Downloading: " + link.url)
        br.follow_link(link)
        material = str(br.geturl()).split('/')[-2:]
        br.retrieve(br.geturl(), outdir + ''.join(material))
        br.back()
