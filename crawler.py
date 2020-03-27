# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
from os import listdir
import time
import csv
import datetime
import operator
import cPickle as pickle
import numpy as np
import requests
import bs4
import re
import urllib
import io
import datetime
import random



def crawler(base_url, htmlFolder, outFile):
    try:
        r = requests.get(base_url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'})

        # 200 is the HTTP status code for "OK", a successful response.
        if r.status_code == 200:
            file_contents = r.content
            write_html_contents = open(htmlFolder + outFile, 'wb')
            write_html_contents.write(file_contents)
            write_html_contents.close()
            return True
        else:
            return False
    except:
        pass

    return False

def run_crawler(d_doionly_d_doi2doi3pmid2pmid3, htmlFolder):

    numToDownload = len(d_doionly_d_doi2doi3pmid2pmid3) * 2   # pub2 and pub3
    cntDone = 0

    for doionly in d_doionly_d_doi2doi3pmid2pmid3:
        # -- pub 2
        doi2 = d_doionly_d_doi2doi3pmid2pmid3[doionly]['doi2']
        base_url = "http://cochranelibrary.com/cdsr/doi/{}/full".format(doi2)
        outFile = doi2 + '.html'
        outFile = outFile.replace('/','_')
        if not check_exist(htmlFolder + outFile):
            print ('..Retrieving {0}'.format(base_url)),

            if crawler(base_url, htmlFolder, outFile):
                print ('..done. '),
            else:
                print ('......failed. '),

            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('{0} of {1} to go.'.format(cntTogo, numToDownload))

        else:
            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('.....{0} exists. {1} of {2} to go.'.format(outFile, cntTogo, numToDownload))

        time.sleep(5)

        doi2 = d_doionly_d_doi2doi3pmid2pmid3[doionly]['doi2']
        base_url = "http://cochranelibrary.com/cdsr/doi/{}/references".format(doi2)
        outFile = doi2 + '_ref.html'
        outFile = outFile.replace('/','_')
        if not check_exist(htmlFolder + outFile):
            print ('..Retrieving {0}'.format(base_url)),

            if crawler(base_url, htmlFolder, outFile):
                print ('..done. '),
            else:
                print ('......failed. '),

            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('{0} of {1} to go.'.format(cntTogo, numToDownload))

        else:
            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('.....{0} exists. {1} of {2} to go.'.format(outFile, cntTogo, numToDownload))

        time.sleep(5)



        # -- pub 3
        doi3 = d_doionly_d_doi2doi3pmid2pmid3[doionly]['doi3']
        base_url = "http://cochranelibrary.com/cdsr/doi/{}/full".format(doi3)
        outFile = doi3 + '.html'
        outFile = outFile.replace('/', '_')
        if not check_exist(htmlFolder + outFile):
            print ('..Retrieving {0}'.format(base_url)),

            if crawler(base_url, htmlFolder, outFile):
                print ('..done. '),
            else:
                print ('......failed. '),

            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('{0} of {1} to go.'.format(cntTogo, numToDownload))

        else:
            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('.....{0} exists. {1} of {2} to go.'.format(outFile, cntTogo, numToDownload))

        time.sleep(5)

        doi3 = d_doionly_d_doi2doi3pmid2pmid3[doionly]['doi3']
        base_url = "http://cochranelibrary.com/cdsr/doi/{}/references".format(doi3)
        outFile = doi3 + '_ref.html'
        outFile = outFile.replace('/', '_')
        if not check_exist(htmlFolder + outFile):
            print ('..Retrieving {0}'.format(base_url)),

            if crawler(base_url, htmlFolder, outFile):
                print ('..done. '),
            else:
                print ('......failed. '),

            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('{0} of {1} to go.'.format(cntTogo, numToDownload))

        else:
            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('.....{0} exists. {1} of {2} to go.'.format(outFile, cntTogo, numToDownload))

        time.sleep(5)

        base_url = "http://cochranelibrary.com/cdsr/doi/{}/information".format(doi3)
        outFile = doi3 + '_info.html'
        outFile = outFile.replace('/', '_')
        if not check_exist(htmlFolder + outFile):
            print ('..Retrieving {0}'.format(base_url)),

            if crawler(base_url, htmlFolder, outFile):
                print ('..done. '),
            else:
                print ('......failed. '),

            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('{0} of {1} to go.'.format(cntTogo, numToDownload))

        else:
            cntDone += 1
            cntTogo = numToDownload - cntDone
            print ('.....{0} exists. {1} of {2} to go.'.format(outFile, cntTogo, numToDownload))

        time.sleep(5)

def get_pub3_with_pub2(inFile):
    f = open(inFile, 'rb')
    reader = csv.reader(f)
    cntRow = -1

    d_doi_pmids = {}
    for col in reader:
        cntRow += 1

        if cntRow == 0:       # header
            continue

        pmid = col[1]
        doi = col[3]

        if doi in d_doi_pmids:
            print ('Double doi')
            sys.exit()
        else:
            d_doi_pmids[doi] = pmid

    d_doionly_d_doi2doi3pmid2pmid3 = {}

    l_doi = d_doi_pmids.keys()
    for i in xrange(len(l_doi)):
        doi = l_doi[i]

        doionly = doi.replace('.pub', '-')
        doionly = doionly.split('-')[0]

        if doionly not in d_doionly_d_doi2doi3pmid2pmid3:

            if (doi[-1] == '2' and doi[:-1] + '3' in d_doi_pmids):
                doi2 = doi
                pmid2 = d_doi_pmids[doi2]
                doi3 = doi[:-1] + '3'
                pmid3 = d_doi_pmids[doi3]

                d_doionly_d_doi2doi3pmid2pmid3[doionly] = {'doi2': doi2, 'pmid2': pmid2, 'doi3': doi3, 'pmid3': pmid3}

            elif (doi[-1] == '3' and doi[:-1] + '2' in d_doi_pmids):
                doi3 = doi
                pmid3 = d_doi_pmids[doi3]
                doi2 = doi[:-1] + '2'
                pmid2 = d_doi_pmids[doi2]

                d_doionly_d_doi2doi3pmid2pmid3[doionly] = {'doi2': doi2, 'pmid2': pmid2, 'doi3': doi3, 'pmid3': pmid3}

    return d_doionly_d_doi2doi3pmid2pmid3



def remove_spchar(text):
    return re.sub(r'[^\x00-\x7f]',r' ',text)

def check_exist_dir(dirname):
    return os.path.isdir(dirname)

def check_exist(fname):
    return os.path.isfile(fname)

def dump_var(data, outFile):
    pickle.dump(data, open(outFile + '.cpickle', "wb"), protocol=2)

def load_var(inFile):
    return pickle.load(open(inFile + '.cpickle', "rb"))

def done():
    print ('\nFinish')
    sys.exit()


if __name__ == '__main__':

    resultFolder = 'Results/'
    cpickleFolder = 'cpickle/'
    dataFolder = 'Datasets/'
    htmlFolder = 'HTML_SystematicReviews/'

    yourFolder = ''
    while yourFolder.strip() == '':
        yourFolder = raw_input("> Enter your folder name: ")

    htmlFolder = yourFolder + '/' + htmlFolder

    if not check_exist_dir(htmlFolder):
        print ('{0} does not exist. Please refer to README file.'.format(htmlFolder))
        sys.exit()

    inFile = dataFolder + 'DOI.csv'
    print ('> Read {0}...')
    d_doionly_d_doi2doi3pmid2pmid3 = get_pub3_with_pub2(inFile)

    print ('> Downloading from Cochrane...')
    run_crawler(d_doionly_d_doi2doi3pmid2pmid3, htmlFolder)

    done()