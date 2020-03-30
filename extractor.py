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
import random
import rules





# ======================================================
# Extract all features from all SRs
# ======================================================
def get_trials_with_participants(d_trialix_tup_authoryear_targettext):

    d_trialix_numParticipants = {}
    for trialix in d_trialix_tup_authoryear_targettext:
        targetText = d_trialix_tup_authoryear_targettext[trialix][1]
        targetText = remove_spchar(targetText)

        numParticipantsSearch, FOUND = rules.get_numbparticipants(targetText)

        if FOUND:
            d_trialix_numParticipants[trialix] = numParticipantsSearch

    return d_trialix_numParticipants

def get_all_trials_information(soup_ref):
    d_trialix_tup_authoryear_targettext = {}

    try:
        sections = soup_ref.find("section", {"class": "characteristicIncludedStudiesContent"})
        tables = sections.find_all("table")

        for ix in xrange(len(tables)):
            trial = tables[ix]
            author_year = ''
            try:
                author_year = str(trial.find("span", {"class": "table-title"}).text).strip()
            except:
                pass

            all_second_rows = trial.find_all('tr')[1]
            participant_column = all_second_rows.findAll('td')[1].text

            d_trialix_tup_authoryear_targettext[ix] = (author_year, participant_column)

    except:
        pass

    return d_trialix_tup_authoryear_targettext

# ======================================================
#      Conclusion
# Output: conclusion_change_st = '' (empty string) if failed, '1' if changed, '0' if not changed
# ======================================================
def extract_conclusion(soup, soup_info, date_publication):
    conclusion_change_st = ''

    try:
        navigation_history = soup.find_all("li", {"class": "cdsr-nav-link article-section-link"})
        for i, v in enumerate(navigation_history):
            if "History" in v.getText():
                conclusion_change_st = rules.get_conclusion(soup_info, date_publication)
                break
    except:
        pass

    return conclusion_change_st

# ======================================================
#      Publication date
# Output: publicationDate_st = '' (empty string) or 'DD MMMM YYYY'
# ======================================================
def extract_datepublication(soup, soup_info):
    publicationDate_st = ''

    try:
        navigation_history = soup.find_all("li", {"class": "cdsr-nav-link article-section-link"})
        for i, v in enumerate(navigation_history):
            if "History" in v.getText():
                publicationDate_st = rules.get_datepublication(soup_info)
                break
    except:
        pass

    return publicationDate_st

# ======================================================
#      Search date
# Output: searchDate_st = '' (empty string) or 'DD MMMM YYYY'
# ======================================================
def extract_searchdate(soup):
    searchDate_st = ''

    try:
        search_date_section = soup.find("div", {"class": "abstract full_abstract"})

        children = search_date_section.findChildren()
        for child in children:
            if child.text == "Search methods" and child.name == "h3":
                child_id = child.get("id")
                searchdate_section = soup.find("section", {"id": child_id})
                searchdate_text = searchdate_section.getText()
                searchDate_st = rules.get_searchDate_rule(searchdate_text)
                break
    except:
        pass

    return searchDate_st



def extract_features_all(htmlFolder, d_doionly_d_doi2doi3pmid2pmid3, outFile):

    st = 'doi\t' \
         'Total trials pub2\tTotal numb participants pub2\tSearch date pub 2\t' \
         'Total trials pub3\tTotal numb participants pub3\tSearch date pub 3\tConclusion\n'
    cnt = 1
    for doionly in d_doionly_d_doi2doi3pmid2pmid3:
        print ('doi: {0}. {1} of {2}'.format(doionly, cnt, len(d_doionly_d_doi2doi3pmid2pmid3)))
        cnt += 1

        doi2 = d_doionly_d_doi2doi3pmid2pmid3[doionly]['doi2']
        inFile2 = doi2 + '.html'
        inFile2 = inFile2.replace('/','_')

        inFile2_ref = doi2 + '_ref.html'
        inFile2_ref = inFile2_ref.replace('/','_')

        doi3 = d_doionly_d_doi2doi3pmid2pmid3[doionly]['doi3']
        inFile3 = doi3 + '.html'
        inFile3 = inFile3.replace('/','_')

        inFile3_ref = doi3 + '_ref.html'
        inFile3_ref = inFile3_ref.replace('/','_')

        inFile3_info = doi3 + '_info.html'
        inFile3_info = inFile3_info.replace('/','_')


        # -- Check if both .pub2 and .pub3 exists and the references and information, only continue if all exist
        if check_exist(htmlFolder + inFile2) and check_exist(htmlFolder + inFile2_ref) and \
        check_exist(htmlFolder + inFile3) and check_exist(htmlFolder + inFile3_ref) and check_exist(htmlFolder + inFile3_info):

            # --------------
            # .pub2
            # --------------
            contents_2 = open(htmlFolder + inFile2, 'r')
            source_code_2 = contents_2.read()
            soup_2 = bs4.BeautifulSoup(source_code_2, 'html.parser')

            contents_ref_2 = open(htmlFolder + inFile2_ref, 'r')
            source_code_ref_2 = contents_ref_2.read()
            soup_ref_2 = bs4.BeautifulSoup(source_code_ref_2, 'html.parser')

            # --------------
            # .pub3
            # --------------
            contents_3 = open(htmlFolder + inFile3, 'r')
            source_code_3 = contents_3.read()
            soup_3 = bs4.BeautifulSoup(source_code_3, 'html.parser')

            contents_ref_3 = open(htmlFolder + inFile3_ref, 'r')
            source_code_ref_3 = contents_ref_3.read()
            soup_ref_3 = bs4.BeautifulSoup(source_code_ref_3, 'html.parser')

            contents_info_3 = open(htmlFolder + inFile3_info, 'r')
            source_code_info_3 = contents_info_3.read()
            soup_info_3 = bs4.BeautifulSoup(source_code_info_3, 'html.parser')

            # --- Search date
            searchDate_st_2 = extract_searchdate(soup_2)
            searchDate_st_3 = extract_searchdate(soup_3)

            if searchDate_st_2 == '' or searchDate_st_3 == '':
                continue

            # --- Publication date
            publicationDate_st_3 = extract_datepublication(soup_3, soup_info_3)

            if publicationDate_st_3 == '':
                continue

            searchDate_dt_3 = datetime.datetime.strptime(searchDate_st_3, '%d %B %Y')
            pubDate_dt_3 = datetime.datetime.strptime(publicationDate_st_3, '%d %B %Y')
            if pubDate_dt_3 < searchDate_dt_3:
                continue


            # --- Conclusion
            date_publication_3 = datetime.datetime.strptime(publicationDate_st_3, '%d %B %Y')
            conclusion_change_st_3 = extract_conclusion(soup_3, soup_info_3, date_publication_3)

            if conclusion_change_st_3 == '':
                continue

            # --- Numb of participants
            d_trialix_tup_authoryear_targettext_2 = get_all_trials_information(soup_ref_2)
            d_trialix_numParticipants_2 = get_trials_with_participants(d_trialix_tup_authoryear_targettext_2)
            enoughTrials_2 = len(d_trialix_numParticipants_2) >= (len(d_trialix_tup_authoryear_targettext_2) / 2.0)
            notzeroTrials_2 = len(d_trialix_numParticipants_2) > 0 and len(d_trialix_numParticipants_2) > 0

            d_trialix_tup_authoryear_targettext_3 = get_all_trials_information(soup_ref_3)
            d_trialix_numParticipants_3 = get_trials_with_participants(d_trialix_tup_authoryear_targettext_3)
            enoughTrials_3 = len(d_trialix_numParticipants_3) >= (len(d_trialix_tup_authoryear_targettext_3) / 2.0)
            notzeroTrials_3 = len(d_trialix_numParticipants_3) > 0 and len(d_trialix_numParticipants_3) > 0

            # --- Only include SRs with half of trials have participants
            if enoughTrials_2 and enoughTrials_3 and notzeroTrials_2 and notzeroTrials_3:
                numTrials_2 = len(d_trialix_numParticipants_2)
                numParticipants_total_2 = sum(d_trialix_numParticipants_2.values())
                numTrials_3 = len(d_trialix_numParticipants_3)
                numParticipants_total_3 = sum(d_trialix_numParticipants_3.values())

                # doi,
                # total trials 2, total numb participants 2, search date 2,
                # total trial 3, total numb participants 3, search date 3
                # conclusion
                st += doionly + '\t' + \
                      str(numTrials_2) + '\t' + str(numParticipants_total_2) + '\t' + searchDate_st_2 + '\t' + \
                      str(numTrials_3) + '\t' + str(numParticipants_total_3) + '\t' + searchDate_st_3 + '\t' + \
                      conclusion_change_st_3 + '\n'

    write_to_file(outFile, st, 'w')

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

def write_to_file(outFile, text, mode):
    with open(outFile, mode) as oF:
        oF.write(text)

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
    resultFolder = yourFolder + '/' + resultFolder

    if not check_exist_dir(htmlFolder):
        print ('{0} does not exist. Please refer to README file.'.format(htmlFolder))
        sys.exit()

    if not check_exist_dir(resultFolder):
        print ('{0} does not exist. Please refer to README file.'.format(resultFolder))
        sys.exit()

    inFile = dataFolder + 'DOI.csv'
    d_doionly_d_doi2doi3pmid2pmid3 = get_pub3_with_pub2(inFile)

    outFile = resultFolder + 'extracted_info.txt'
    extract_features_all(htmlFolder, d_doionly_d_doi2doi3pmid2pmid3, outFile)

    done()