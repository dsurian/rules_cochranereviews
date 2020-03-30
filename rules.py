# -*- coding: utf-8 -*-
from __future__ import division
import sys
import time
import datetime
import operator
import requests
import bs4
import re
import urllib
import io
import datetime


# ======================================================
#      Search date
# ======================================================
def get_searchDate_rule(st):
    st = st.replace(',',' ')
    st = st.replace('.',' ')

    # -- "DD MM YYYY" format or  "MM YYYY" format (without date) or "MM DD YYYY" format
    # ----------------------------------------------------------------------------------
    pattern_date = re.compile(r"""
        (\d+)*                            # date, one or more of any digit [0-9], zero or more times
        (\s+)*                            # one or more of whitespaces, zero or more times
        (                                 # month, should exist
            Jan|January
            |Feb|February
            |Mar|March
            |Apr|April
            |May
            |Jun|June
            |Jul|July
            |Aug|August
            |Sep|September
            |Oct|October
            |Nov|November
            |Dec|December
        )                     
        (\s+)*                            # one or more of whitespaces, zero or more occurrences
        (\d+)*                            # date, one or more of any digit [0-9], zero or more times
        (\s+)*                            # one or more of whitespaces, zero or more occurrences
        (\b18\d{2}|19\d{2}|2\d{3}\b)      # year, 1800-2999
    """, re.VERBOSE | re.IGNORECASE)

    # -- Rule 1
    # ----------
    rule = pattern_date.findall(st)
    l_datest = rule

    l_datedt = []
    for s_dt in l_datest:
        datest = ' '.join(list(s_dt))

        # -- Replace any redundant whitespaces to a single whitespace
        datest_clean = re.sub(r'\s+', ' ', datest)
        datest_clean = datest_clean.strip()
        l_datestclean = datest_clean.split()

        if len(l_datestclean) == 2:     # only month and year exist
            datest_clean = datest_clean.replace(' ', ' 1 ')   # replace with the first date of the month, if not exists

        # -- Convert to datetime format
        try:
            datest_clean = datetime.datetime.strptime(datest_clean, '%d %B %Y')    # DD MMMM YYYY, e.g. 21 March 2013
            l_datedt.append(datest_clean)
        except:
            pass

        try:
            datest_clean = datetime.datetime.strptime(datest_clean, '%d %b %Y')    # DD MMM YYYY, e.g. 21 Mar 2013
            l_datedt.append(datest_clean)
        except:
            pass

        try:
            datest_clean = datetime.datetime.strptime(datest_clean, '%B %d %Y')    # MMMM DD YYYY, e.g. March 21 2013
            l_datedt.append(datest_clean)
        except:
            pass

        try:
            datest_clean = datetime.datetime.strptime(datest_clean, '%b %d %Y')    # MMM DD YYYY, e.g. Mar 21 2013
            l_datedt.append(datest_clean)
        except:
            pass

    # -- Get the latest date
    searchDate_latest_st = ''
    if len(l_datedt) > 0:
        l_datedt.sort()
        searchDate_latest_st = l_datedt[len(l_datedt)-1]

        # -- Convert to string in a uniform format: DD MMMM YYYY
        searchDate_latest_st = searchDate_latest_st.strftime('%d %B %Y')

    searchDate_latest_st = searchDate_latest_st.lstrip('0')

    return searchDate_latest_st

# ======================================================
#      Date of publication
# ======================================================
def get_datepublication(soup_info):
    publication_date_st = ''

    sections_publication = soup_info.find_all("section", {"id": "information"})
    for i, v in enumerate(sections_publication):
        pub_date = v.find("span", {"class": "publish-date"})
        publication_date = pub_date.text

        # -- Rule 1
        # ----------
        pattern_date = re.compile("""
            (\d+.*?\d+)          # find any digits followed by anything (whitespace, characters, etc.) and digits
        """, re.VERBOSE | re.IGNORECASE)

        rule = pattern_date.search(publication_date)


        extract_date = rule

        publication_date_st = str(extract_date.group())
        break

    date_publication = datetime.datetime.strptime(publication_date_st, '%d %B %Y')  # string to date

    # -- Convert to string in a uniform format: DD MMMM YYYY
    date_publication_st = date_publication.strftime('%d %B %Y')

    return date_publication_st

# ======================================================
#      Conclusion
# ======================================================
def get_conclusion(soup_info, date_publication):
    list_history = []
    sections_history = soup_info.find_all("section", {"class": "history"})
    for i, v in enumerate(sections_history):
        tables_history1 = v.find("div", {"class": "table"})
        tables_history = tables_history1.find("table")

        for his_tab_row in tables_history.find_all("tr")[1:]:
            save_history = his_tab_row.text
            list_history.append(save_history)

    sections_whats_new = soup_info.find_all("section", {"class": "whatsNew"})
    for a, b in enumerate(sections_whats_new):
        tables_whatsnew = b.find("table")

        for whatsnew_tab_row in tables_whatsnew.find_all("tr")[1:]:
            list_history.append(whatsnew_tab_row.text)

    # -- Rule 1
    # ------------
    list_con = []
    for i in list_history:
        txt = str(i)
        match_pattern = re.compile(r"""
            conclu\w+\s+.*?\s{2}          # find word 'conclu' followed by one or more of any alphanumeric character  
        """, re.VERBOSE | re.IGNORECASE)
        match_search = match_pattern.search(txt)

        if match_search:
            list_con.append(txt)

    l_dateconclusion = []
    l_conclusion = []
    d_ix_diffdays = {}
    ix = 0
    for con in list_con:
        # -- Extract rows discussing about conclusion
        # -----------
        match_pattern = re.compile(r"""
            conclu\w+\s+.*?\s{2}          # find word 'conclu' followed by one or more of any alphanumeric character  
        """, re.VERBOSE | re.IGNORECASE)
        match_search = match_pattern.search(con)

        if match_search:
            conclusion = match_search.group().rstrip()
            conclusion_st = str(conclusion).strip('[]')

            # -- Extract corresponding dates where something is said about conclusion
            # -----------
            match_pattern = re.compile(r"""
                (\d+\s{0,1}\w+\s{0,1}\d+)    # any digit (1 or more), any whitespace character, any alphanumeric characters, any whitespace character, and any digit (1 or more) 
            """, re.VERBOSE | re.IGNORECASE)
            match_search = match_pattern.search(con)

            if match_search:
                conclusion_date_st = match_search.group().rstrip()
                date_conclusion = datetime.datetime.strptime(conclusion_date_st, '%d %B %Y')

                diff_days = (date_publication - date_conclusion).days

                if diff_days > 0 and conclusion_date_st != '':
                    l_dateconclusion.append(date_conclusion)
                    l_conclusion.append(conclusion_st)
                    d_ix_diffdays[ix] = diff_days
                    ix += 1

    ltup_ix_diffdays = sorted(d_ix_diffdays.items(), key=operator.itemgetter(1), reverse=False)
    ix_closest = ltup_ix_diffdays[0][0]
    dateconclusion_current = l_dateconclusion[ix_closest]
    conclusionst_current = l_conclusion[ix_closest]

    conclusion_change = -1
    match_pattern = re.compile(r"""
        (not|no|unchang\w+)       # find word 'not' or 'no' 
    """, re.VERBOSE | re.IGNORECASE)
    match_search = match_pattern.search(conclusionst_current)

    rule = match_search

    if rule:
        conclusion_change = 0
    else:
        conclusion_change = 1


    conclusion_change_st = str(conclusion_change)

    return conclusion_change_st

# ======================================================
#      Number of participants
# ======================================================
# -- Remove comma that separate thousands
def clean_commathousand(targetText):
    pat_commathousand = re.compile(r"""
            \d+,\d+    # Find pattern started with digits followed by comma and digits
        """, re.VERBOSE | re.IGNORECASE)
    search_commathousand = pat_commathousand.search(targetText)

    if search_commathousand:
        res = search_commathousand.group()

        sPart = res.split(',')
        if len(sPart[0]) <= 3 and len(sPart[1]) == 3:
            res_clean = res.replace(',', '')
            targetText = targetText.replace(res, res_clean)
        else:
            res_clean = res.replace(',', ' ')
            targetText = targetText.replace(res, res_clean)

    return targetText

def patterns(targetText):
    # -- Find pattern started with 'overall', if exists, followed by ':', digits, any combinations of characters,
    #    and any of defined words
    pat_being = re.compile(r"""
        (overall)?\s*(:)?\s*\d+\s+[a-zA-Z]*\s*(participants|[^a-zA-Z]men[^a-zA-Z]|women|enrolled|patients|people|adolescents|adults|smokers|inoculated|children|infants|neonates|workers)(with)?
    """, re.VERBOSE | re.IGNORECASE)
    search_being = pat_being.search(targetText)

    pat_n = re.compile(r"""
        [^a-zA-Z]n\s*(=|:)\s*\d+        # Find pattern with 'n = digit' or 'n : digit' with or without space in between
    """, re.VERBOSE | re.IGNORECASE)
    search_n = pat_n.search(targetText)

    # -- Find pattern started with 'number' followed by defined word and digits
    pat_nrandomised = re.compile(r"""
        (number)?\s+(randomised|randomized)\s*(:)\s+\w{0,1}\s*\d+
    """, re.VERBOSE | re.IGNORECASE)
    search_nrandomised = pat_nrandomised.search(targetText)

    # -- Find pattern started with digits followed by word 'were' or 'patients', if exists, followed by word
    #    'randomised' or 'randomized
    pat_wererandomised = re.compile(r"""
        \d+\s+(were|patients)?\s*(randomised|randomized)
    """, re.VERBOSE | re.IGNORECASE)
    search_wererandomised = pat_wererandomised.search(targetText)

    pat_nbr = re.compile(r"""
        (number|total)\s*\w*(:)\s+\d+  # Find pattern started with 'number' or 'total' followed by ':' and digits
    """, re.VERBOSE | re.IGNORECASE)
    search_nbr = pat_nbr.search(targetText)

    pat_rndly = re.compile(r"""
        \d+\s+\w*\s*(randomly)   # Find pattern started with digits followed by word 'randomly'
    """, re.VERBOSE | re.IGNORECASE)
    search_rndly = pat_rndly.search(targetText)

    # -- Find pattern with 'n = digits' (with or without spaces in between, with or without parenthesis)
    pat_multiplen = re.compile(r"""
        (\s+|[^a-zA-Z]|\()(n)\s*(=|:)\s*(\d+)\s*
    """, re.VERBOSE | re.IGNORECASE)
    findall_multiplen = pat_multiplen.findall(targetText)

    pat_totalnumber = re.compile(r"""
        (total)\s+(number)   # Find pattern started with 'total' followed by 'number'
    """, re.VERBOSE | re.IGNORECASE)
    search_totalnumber = pat_totalnumber.search(targetText)

    # -- Find pattern started with 'total number of patients' followed by '=' or ':' and digits
    pat_totalnumberofpatients = re.compile(r"""
        (total)\s+(number)\s+(of)\s+(patients)(:|=)\s+\d+
    """, re.VERBOSE | re.IGNORECASE)
    search_totalnumberofpatients = pat_totalnumberofpatients.search(targetText)

    # -- Find pattern started with 'overall', if exists, followed by ':', digits and any of defined words
    #    (different from pat_being)
    pat_being_2 = re.compile(r"""
        (overall)?\s*(:)?\s*\d+\s+(\w+\s+)*(participants|[^a-zA-Z]men[^a-zA-Z]|women|enrolled|patients|people|adolescents|adults|smokers|inoculated|children|infants|neonates|workers)(with)?
    """, re.VERBOSE | re.IGNORECASE)
    search_being_2 = pat_being_2.search(targetText)

    # -- Find pattern started with words 'male/female' followed by ':' or '=' and digits/digits
    pat_malefemale = re.compile(r"""
        (male)(\/)(female)(:|=)\s*(\d+)(\/)(\d+)
    """, re.VERBOSE | re.IGNORECASE)
    findall_malefemale = pat_malefemale.findall(targetText)

    pat_total = re.compile(r"""
        (total)\s+(\w+)?\d+    # Find pattern started with 'total' and digits
    """, re.VERBOSE | re.IGNORECASE)
    search_total = pat_total.search(targetText)

    # -- Find pattern started with digits followed by 'participants were randomly assigned'
    #    followed by ':' or '=' and digits
    pat_wererandomlyassigned = re.compile(r"""
        \d+\s+(participants)\s+(were)\s+(randomly)\s+(assigned)\s*(:|=)\s*\d+
    """, re.VERBOSE | re.IGNORECASE)
    search_wererandomlyassigned = pat_wererandomlyassigned.search(targetText)

    # -- Find pattern started with 'sample size' followed by ':' or '=' and digits
    pat_samplesize = re.compile(r"""
        (sample)\s+(size)\s*(:|=)\s*\d+
    """, re.VERBOSE | re.IGNORECASE)
    search_samplesize = pat_samplesize.search(targetText)

    # -- Find pattern started with digits followed by any defined word and 'were enrolled'
    pat_wereenrolled = re.compile(r"""
        \d+\s+(families|participants|patients)\s+(were)\s+(enrolled)
    """, re.VERBOSE | re.IGNORECASE)
    search_wereenrolled = pat_wereenrolled.search(targetText)

    # -- Find pattern started with 'number of participants' followed by anything, if exists, and ':' with digits
    pat_nbrofparticipants = re.compile(r"""
        number\s+of\s+participants(.*)(:)\s*\d+
    """, re.VERBOSE | re.IGNORECASE)
    search_nbrofparticipants = pat_nbrofparticipants.search(targetText)

    # -- Find pattern started with digits followed by anything then any defined word
    pat_being_3 = re.compile(r"""
        \d+\s+.*(participants|[^a-zA-Z]men[^a-zA-Z]|women|enrolled|patients|people|adolescents|adults|smokers|inoculated|children|infants|neonates|workers)(with)?
    """, re.VERBOSE | re.IGNORECASE)
    search_being_3 = pat_being_3.search(targetText)

    return search_being, search_n, search_nrandomised, search_wererandomised, \
           search_nbr, search_rndly, findall_multiplen, search_totalnumber, \
           search_totalnumberofpatients, search_being_2, findall_malefemale, \
           search_total, search_wererandomlyassigned, search_samplesize, \
           search_wereenrolled, search_nbrofparticipants, search_being_3

def get_numbparticipants(targetText):
    targetText = clean_commathousand(targetText)

    search_being, search_n, search_nrandomised, search_wererandomised, \
    search_nbr, search_rndly, findall_multiplen, search_totalnumber, \
    search_totalnumberofpatients, search_being_2, findall_malefemale, \
    search_total, search_wererandomlyassigned, search_samplesize, \
    search_wereenrolled, search_nbrofparticipants, search_being_3 = patterns(targetText)


    rule_1 = search_nrandomised and search_being
    rule_2 = search_wererandomised
    rule_3 = search_totalnumberofpatients
    rule_4 = search_totalnumber and search_being
    rule_5 = search_nbr and search_being
    rule_6 = search_n and search_being and search_total == None
    rule_7 = search_n and len(findall_multiplen) < 2
    rule_8 = len(findall_multiplen) > 1
    rule_9 = search_nbr
    rule_10 = search_rndly
    rule_11 = search_nrandomised
    rule_12 = search_being

    rule_13 = search_wererandomised and search_being
    rule_14 = search_wererandomlyassigned
    rule_15 = search_nbrofparticipants
    rule_16 = search_total
    rule_17 = len(findall_malefemale) > 0
    rule_18 = search_being_2

    rule_19 = search_samplesize and search_being
    rule_20 = search_wereenrolled
    rule_21 = search_samplesize
    rule_22 = search_being_3

    numParticipantsSearch = -1
    FOUND = False


    if rule_1:
        start_nrandomised = search_nrandomised.start()
        start_being = search_being.start()

        if start_nrandomised < start_being:
            res_st = search_nrandomised.group()

            # --- Extract the number
            m = re.findall(r'\d+', res_st)
            numParticipantsSearch = int(m[0])

            FOUND = True
        else:
            res_st = search_being.group()

            # --- Extract the number
            m = re.findall(r'\d+', res_st)
            numParticipantsSearch = int(m[0])

            FOUND = True

    elif rule_13:
        start_wererandomised = search_wererandomised.start()
        start_being = search_being.start()

        if start_wererandomised < start_being:
            res_st = search_wererandomised.group()

            # --- Extract the number
            m = re.findall(r'\d+', res_st)
            numParticipantsSearch = int(m[0])

            FOUND = True
        else:
            res_st = search_being.group()

            # --- Extract the number
            m = re.findall(r'\d+', res_st)
            numParticipantsSearch = int(m[0])

            FOUND = True

    elif rule_2:
        res_st = search_wererandomised.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_14:
        res_st = search_wererandomlyassigned.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_3:
        res_st = search_totalnumberofpatients.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_15:
        res_st = search_nbrofparticipants.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_16:
        res_st = search_total.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_4:
        res_st = search_being.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_5:
        res_st = search_being.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_6:
        start_n = search_n.start()
        start_being = search_being.start()

        if start_n < start_being:
            res_st = search_n.group()

            # --- Extract the number
            m = re.findall(r'\d+', res_st)
            numParticipantsSearch = int(m[0])

            FOUND = True
        else:
            res_st = search_being.group()

            # --- Extract the number
            m = re.findall(r'\d+', res_st)
            numParticipantsSearch = int(m[0])

            FOUND = True

    elif rule_7:  # 1st     # 50
        res_st = search_n.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_19:
        start_samplesize = search_samplesize.start()
        start_being = search_being.start()

        if start_samplesize < start_being:
            res_st = search_samplesize.group()

            # --- Extract the number
            m = re.findall(r'\d+', res_st)
            numParticipantsSearch = int(m[0])

            FOUND = True

        else:
            res_st = search_being.group()

            # --- Extract the number
            m = re.findall(r'\d+', res_st)
            numParticipantsSearch = int(m[0])

            FOUND = True

    elif rule_8:
        n = 0
        for s in findall_multiplen:
            n += int(s[len(s)-1].strip())
        numParticipantsSearch = n

        FOUND = True

    elif rule_9:
        res_st = search_nbr.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_10:
        res_st = search_rndly.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_11:
        res_st = search_nrandomised.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_20:
        res_st = search_wereenrolled.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_17:
        s = findall_malefemale[0]
        numParticipantsSearch = int(s[4].strip()) + int(s[6].strip())

        FOUND = True

    elif rule_21:
        res_st = search_samplesize.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_12:
        res_st = search_being.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_18:
        res_st = search_being_2.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    elif rule_22:
        res_st = search_being_3.group()

        # --- Extract the number
        m = re.findall(r'\d+', res_st)
        numParticipantsSearch = int(m[0])

        FOUND = True

    return numParticipantsSearch, FOUND


