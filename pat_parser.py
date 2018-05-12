#pat_parser.py
"""A sandbox parser for my true love."""

from __future__ import with_statement, generators, print_function, division, unicode_literals
import pandas as pd
from numpy import NaN
from time import time


fn = "Columbia Run 1_r.xlsx"
sheet = "4W80"

df = pd.read_excel(fn, parse_cols= 25)

x = 1








sdict = \
    {'sfn': "01 Deliverable - Drucker Institute - Analysis - 20170613.xlsx",
     'tab': "No of inventions",
     'var_name' : "Number_of_Inventions_{}",  #wave4; Rate_of_Abandonment
     'col_renames': {'Company Name': coname_col, 'GICS Industry': industry_col,},
     'scol': [coname_col, 'drop_me', 'Industry', 'Count', 'PS_Pay_Percent_Diff_{}'],
     'skiprows': 1,   #rows to skip before data
     'sdc1': 2,  #first source data col
     'ddc1': 2,  #first destination data col
     'junkcols': ['Company Ticker'],   #cols to drop; neither headers like FirmID nor data for retention
     }
s = sdict
get_firmid_from_vendor_name

def pat_parse():
	"""Parse data, scrapping unneeded columns, adding FirmID, cleaning text.

	    Reads spreadsheet cf. settings.py.  Returns a DataFrame"""
	fn = abs_path(sdict['sfn'], path)

	#read the data
	skiprows = None
	#if 'skiprows' in sdict: skiprows = sdict['skiprows']
	raw = pd.read_excel(fn,  skiprows= skiprows,  header = None)

	#clean up        
	cols =  raw.loc[0].combine_first(raw.loc[1]).copy()  #combines split header rows
	raw.columns = cols

	raw.drop(s['junkcols'], inplace = True,              #loses any intersperced cols we don't need
		     axis = 1)             
	raw = raw.iloc[2:, :-2]                              #drops the header and last 2 cols
	raw.replace(missing, 0, inplace = True)

	raw = raw.astype('float64',                          #convert any col we can to a numeric type
		             errors = 'ignore')
	raw = fix_non_ascii_df(raw)                          #fix non-ascii chars in text

	raw.columns.rename(s['col_renames'], inplace = True) #standardize company, industry names

	raw.insert(0, firmid_col,                            #look up and insert FirmID
		       get_firmid_from_vendor_name(raw[coname_col]) )

	aggregate
	"""
    company, ticker, GICS industry, inventions by year, sum(inventions by year), CAGR

             tabs:  ["No of inventions", Quality of patents", "Rate of abandonment", "R&D spending"]

            need:
            - replace "-" with 0 (not NaN)
            - get FirmID
            - good ind averages    """
	a = 1