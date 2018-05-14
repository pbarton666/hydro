#scratch_script.py
"""Python scratch file for scripts-in-process"""

import unittest
import numpy
import pandas as pd
import os
from collections import namedtuple
from fuzzywuzzy import fuzz, process

#set the location (e.g., "../current_wave_data/dictionary.xlsx")
DICT_FILE = 'dictionary.xlsx'
dict_path = os.path.join("..", "current_wave_data", DICT_FILE)

#set the sheet (tab) name containing deliberately-excluded firms
EXCLUDED_SHEET = 'Excluded_parents'

def create_company_dict(sheetname = "Name_firmID", parse_cols = "A:C",
                        dict_path = None):
    """Create a dict of CompanyName from the population frame.  Keys
       are the official company name; values are a tuple of firmID, ticker).
    """
    #create a dict of {company name:  (firmID, ticker)}
    company_dict = {}
    company_data = namedtuple("company_data", ('firmID', 'ticker'))
    
    #this reads the tab matching name to firmID and ticker
    df = pd.read_excel(dict_path, sheetname = sheetname,
                       header = 1, parse_cols = parse_cols)

    #the for loop cleans up the data and adds each line to the dict
    for rowid in range(len(df)):
        company_dict[str(df.iloc[rowid]['company']).strip()] = \
            company_data(df.iloc[rowid]['firmID'], str(df.iloc[rowid]['ticker']))
        
    return company_dict

def create_acsi_dict(sheetname = "ACSI", parse_cols = "A:C", 
                     dict_path = None):
    """Create a dict of the ACSI companies mapped to parent and ACSI industry.
       Keys are the ACSI company name; values are a tuple of (ParentCompanyName and Industry).
    """
    #ACSI mappings {ASCI name:  (parent company, ACSI industry)}
    acsi_dict = {}
    acsi_data = namedtuple("acsi_data", ('ParentCompanyName', 'Industry'))
    
    #this reads the tab matching name to firmID and ticker
    df = pd.read_excel(dict_path, sheetname = sheetname,
                       header = 1, parse_cols = parse_cols)
    
    #the for loop cleans up the data and adds each line to the dict
    for rowid in range(len(df)):
        acsi_dict[str(df.iloc[rowid]['CompanyOrBrand']).strip()] = \
            acsi_data(df.iloc[rowid]['ParentCompanyName'], \
                      str(df.iloc[rowid]['Industry']))
    return acsi_dict        

def create_temkin_dict(sheetname = "Temkin", parse_cols = "A:C",
                       dict_path = None):
    """Create a dict of the Temkin companies mapped to parent and Temkin industry.
       Keys are the Temkin company name; values are a tuple of (ParentCompanyName and Industry).
       
       NB:  within the spreadsheet, we need to create a Temkin CompanyOrBrand to disambiguate
       company divisions.  For instance AT&T is the company name for both "Internet Service" and
       "TV Service".  The spreadsheet should have "AT&T_Internet Service" in the CompanyOrBrand
       column. 
    """

    temkin_dict = {}
    temkin_data = namedtuple("temkin_data", ('ParentCompanyName', 'Industry'))
    
    #this reads the tab matching name to firmID and ticker
    df = pd.read_excel(dict_path, sheetname = sheetname,
                       header = 1, parse_cols = parse_cols)
    
    #the for loop cleans up the data and adds each line to the dict
    for rowid in range(len(df)):
        temkin_dict[str(df.iloc[rowid]['CompanyOrBrand'])] = \
            temkin_data(df.iloc[rowid]['ParentCompanyName'], \
                      str(df.iloc[rowid]['Industry'])                      
                      )
    return temkin_dict    

def check_for_problems(vendor_dict, sheetname = None, excluded_sheet_name = EXCLUDED_SHEET):    
    """
      For QA, make sure all the parent companies have a 'matcher' in the main
      population frame.
      
      If no 'matcher' found, check the list of known
      exclusions (Excluded_parent tab of dictionary spreadsheet).print out the companies not
      found."""
    companies_in_pop_frame = company_dict.keys()

    likely_excludes = []
    likely_matchers = []
    #
    df = pd.read_excel(dict_path, sheetname = excluded_sheet_name,
                           header = 1, parse_cols = "A")
    companies_excluded_from_pop_frame = list(df['Excluded'])
    for key in vendor_dict.keys():
        parent_in_vendor_data =  vendor_dict[key].ParentCompanyName
        
        #Is the company in the population frame?
        if not type(parent_in_vendor_data) == int and not parent_in_vendor_data in companies_in_pop_frame:
            #... if it's not, have we explicitly excluded it?
            if not parent_in_vendor_data in companies_excluded_from_pop_frame:
                # if it's not explicitly "in" or "out", try to find a fuzzy match.
                proposed = do_fuzzy_lookup(companies_in_pop_frame, parent_in_vendor_data)
                if proposed:
                    likely_matchers.append(str(parent_in_vendor_data) + "|" + str(proposed))
                else:
                    likely_excludes.append(str(parent_in_vendor_data)+ "|" + str(sheetname))
                    
    if likely_excludes or likely_matchers:
        message = \
        "I found these companies in the ParentCompanyName column in\n the {} tab of {}.\n"
        "Please make the necessary corrections in the {} tab or add them to\n"
        "the Excluded... tab.   The given name and a fuzzy match are provided below:\n"
            
        print(message.format(sheetname, DICT_FILE))
        for likely_matcher in likely_matchers:			
            print(likely_matcher)
        print()
        print("And you may want to add these to the Excluded_parents tab (but check first):\n")
        
        for likely_exclude in likely_excludes:			
            print(likely_exclude)            
        print('*' * 20)
        print()

def do_fuzzy_lookup(using_this = None, item_to_find = None):
    "tries to find a fuzzy match with fuzzywuzzy library"
    finding =  process.extractOne(item_to_find, using_this)
    matcher = finding[0]
    score = finding[1]

    if score > 90:
        return matcher


if __name__ == '__main__':

    #population frame
    sheetname = "Name_firmID"
    company_dict = create_company_dict(sheetname = sheetname, parse_cols= "A:C",
                                       dict_path= dict_path)
    a = 1
    
    #Temkin
    sheetname = "Temkin"
    temkin_dict = create_temkin_dict(sheetname = sheetname, parse_cols= "A:C",
                                     dict_path= dict_path)
    check_for_problems(temkin_dict, sheetname = sheetname, excluded_sheet_name = "Excluded_Temkin")    
    
    #ACSI
    sheetname = "ACSI"
    acsi_dict = create_acsi_dict(sheetname = sheetname, parse_cols= "A:C",
                                       dict_path= dict_path)
    check_for_problems(acsi_dict, sheetname = sheetname, excluded_sheet_name = "Excluded_ACSI")

    a = 1
        
    

    

    a = 1

