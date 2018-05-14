#utilities.py
# -*- coding: utf-8 -*-
from __future__ import print_function

import pandas as pd
import os
import sys
from numpy import nan,  NaN, mean
import csv
import SpssClient as client
from SpssClient import  SpssClientException
from dictionary.py_dictionaries import id_parent, parent_id
from dictionary import py_dictionaries

from general_settings.general_settings import * 

#from settings import *

class FileNotFoundException(Exception):
    pass

class NameNotFoundError(Exception):  
    pass


def average_cols(df = None, cols = None):
    """Averages columns in a DataFrame.  """
    return df[cols].mean(axis = 1)

def compute_t_from_z(df):
    """Transforms Z-scores to  (mean = 50, sd = 10) T-scores.
          Input is a DataFrame, output is  a DataFrame.  Renames
          output col to prepend a 'T'. """
    
    return df * 10 + 50
    new_col_names = []
    
    #text cols
    for col in df.columns[:first_data_col]:
        new_col_names.append(col)
    
    #data cols    
    for col in df.columns[first_data_col:]:
        df[col] = df[col]* 10 + 50
        new_col_names.append('T' + col)
    
    #rename
    df.columns = new_col_names
         
    return df

def poke_in_col(sdf = None, scol = None,
                ddf = None, dcol = None,
                qa_mode = False):

    """Specs include:  source DataFrame  sdf
                       source cols        [scol]
                       destination DataFrame ddf
                       destination cols *after which* to insert the new column [dcol]
                       qa_mode - adds a "_qa" to the destination column name

        The source and destination DataFrames should already be indexed correctly
        
    General-purpose method for updating data tables.    
    """
    
    "The destination col names are optional; use if provided, otherwise use source names."
    if not dcol:
        dcol = scol
        
    #make sure we lave lists
    if not isinstance(scol, list): scol = [scol]
    if not isinstance(dcol, list): dcol = [dcol]
    
    dest_cols = list(ddf.columns)
    
    for sc, dc in zip(scol, dcol):
        #if the selected col is the last one, well do an append
        if dc == dest_cols[-1]:
            if qa_mode:                
                ddf[sc] = sdf[sc]
                ddf.rename(columns = {sc: sc + '_qa'}, inplace = True)
            else:                
                ddf[sc] = sdf[sc]                
        else:
            try:                
                ix = ddf.columns.get_loc(dc)
            except:
                #existing column not found - stick it at the beginning
                ix = -1
            if qa_mode:
                ddf.insert(loc = ix + 1, column = dc + "_qa", value = sdf[sc])
            else:
                ddf.insert(loc = ix + 1, column = sc, value = sdf[sc])
    
    return ddf
        
        
    
    


def compute_one_z(ser):
    return  (ser -ser.mean()) / ser.std()

def compute_z(df, as_tscore = False, first_data_col = 2, append_cols = False):
    """Takes a DataFrame as input.  Computes z-score.   Returns a DataFrame.
          The data is a true z-score or (optionally) a 100-point scaled T-score.
          If append_cols == True, returns original w/ new cols appended, otherwise
          just returns the transformed data.
    """
    #new df
    new = pd.DataFrame()
    if not append_cols:
        #put text cols in new df
        for col in df.columns[:first_data_col]:
            new[col] = df[col]
    
    #process data cols
    for col in df.columns[first_data_col:]:
        z = (df[col] - df[col].mean()) / df[col].std()
        if as_tscore:
            data = compute_t_from_z(z)
            new_name = "T" + str(col)
        else:
            new_name = "Z" + str(col)
            data = z
        if append_cols:
            df[new_name] = data
        else:
            new[new_name] = data
        
    if not append_cols:
        return new
    else:
        return df    
    

def scrub_missing(df = None, missing = None, replace_with = None):
    """Replaces any missing values (N/A, etc.) found in a DataFrame).  Returns
       the clean DataFrame.  missing is some iterable of strings."""
    
    if type(missing) in (str, int, float):
        df.replace(missing, replace_with, inplace = True)
    else:		
        for m in missing:			
            df.replace(m, replace_with, inplace = True)
    return df

def run_syntax(syntax, tag = None, out_log_fn = None, syn_file = None, my_client = None):
    "Starts a SPSSClient process, runs syntax and exits."
    
    #https://www.ibm.com/support/knowledgecenter/en/SSLVMB_sub/
    # statistics_python_plugin_uber_project_ddita/
    # statistics_python_plugin_uber_project_ddita-gentopic1.html
    
    print("running syntax for {}.".format(tag))
    print("using file {}.".format(syn_file))
    
    if my_client:
        client = my_client
    else:
        client.StartClient()
         
    try:
            
        try:            
            client.RunSyntax(syntax)
        except Exception as e:
            print("\nSPSSClientException encountered while running {}:  {}".format(tag, e))
        try:
            out_docs = client.GetOutputDocuments()
            everything = out_docs.GetItemAt(out_docs.Size() - 1)
            if everything:                
                everything.ExportDocument(client.SpssExportSubset.SpssVisible,
                                          out_log_fn,
                                          client.DocExportFormat.SpssFormatText )        
        except Exception as e:
            print("\nSPSSClientException encountered while running {}:  {}".format(tag, e))
        
        print("done.\n")
    
    except:
        client.Exit()
        client.StopClient()
        
def detect_non_ascii(strg) :
    try:        
        strg.decode('ascii')
    except:
        print("{} has a non-ascii character".format(strg))
        
def aggregate_vendors(agg_dict = None, path = None, wave = None,
                      final_product_fn= None, qa_mode = None,
                      dfn = None, ddir = None):
    
    """Accepts input dicts that point to the files / directories  to be
         combined.   As some of these contain both current and historic values, the user can
         further specify which columns of data to import from each.
    
       Quality assurance is accomplished by inserting imported columns (from whatever source)
         into the existing "whole enchilada" aggregation file (e.g., "Raw Customer Satisfaction 2016")
         with <variable>_qa names.  The '_qa' columns are extracted, as are the matching existing columns,
         and stored in separate data files.   The files are then passed into an Excel sheet containing the
         'target' values, the 'actual' values, and first differences.
    
       Key parameters are derived from settings.py, which can be customized for each data aggregation
         step.  From within the aggregation directory e.g., agg_cust_sat, you would have a settings.py
         file, aggregate() would be imported an run for that location.  This keeps its operations atomic.
         """
    
    #some strings for the col names
    wave2 = str(wave)[2:]
    wave4 = str(wave)

    #the destination file (contains historical records, will be augmented)
    ddf = pd.read_csv(os.path.join(path, ddir, dfn ))
    
    
    #The agg_dict runs the show.  	
    for a in agg_dict:

        #create DataFrames for the source and target files
        sdf = pd.read_csv(os.path.join(path, a['sdir'], a['sfn']))

        ##TODO: use general settings firmid col in vendor data pulls.  Temkin has wrong case.
        for df in (sdf, ddf):
            if df.index.name:            
                df.reset_index(inplace = True)
            if firmid_col.lower() in df.columns[0].lower():
                cols = list(df.columns)			
                cols[0] = firmid_col  
                df.columns = cols

            #replaces any blanks w/ nan, then nan to nan_value (e.g., 99999)
            df.replace(' ', nan, inplace = True)
            df.replace(nan, nan_value, inplace = True)
            df.set_index(firmid_col, inplace = True)

        #use an optional desination col name if provided, otherwise use source col name
        if not 'dcol' in a:
            a['dcol'] = a['scol']
        ddf = poke_in_col(sdf = sdf, scol = a['scol'],
                          ddf = ddf, dcol = a['dcol'],
                          qa_mode = qa_mode)

    #write final product as a csv
    final_fn =  os.path.join(path, a['ddir'], final_product_fn)
    ddf.to_csv(final_fn)	

    #The rest of this code just dumps a QA file	
    if qa_mode:

        #figure out 'keepers' (columns that represent current wave data)
        #drop all data columns w/o wave e.g., 16 in them
        keepers = list(ddf.columns[:1])
        #Satmetrix column names are weird .. but we need to use them for the 2016 qa
        for c in ddf.columns:
            if 'Satmetrix' in c:
                if (wave4 in c[:10] and wave2 in c[10:]) or "Firm_Minus" in c:
                    keepers.append(c)
            else:
                if wave2 in c: keepers.append(c)
        ddf = ddf[keepers] 			  

        #sort data cols by column name
        ddf.iloc[:, 1:].sort_index(axis = 1, inplace = True)

        #make target and results dfs for QA spreadsheet in - start with a copy
        dfcp = ddf.copy()
        dfcp.replace(nan, nan_value, inplace = True)

        #slices to manage columns
        nondata_cols = slice(None, 1)
        data_cols = slice(1, None)

        #find all cols with a '_qa'; these will be the results for the qa file
        result_cols = list(dfcp.columns)[nondata_cols]
        result_cols = result_cols + sorted([col for col in dfcp.columns[data_cols] if '_qa' in col])

        #find all the matching columns from the target (the data sans the '_qa' suffix)
        target_cols = list(dfcp.columns[nondata_cols])
        target_cols =  target_cols + sorted([col[:-3] for col in result_cols[data_cols]])

        #grab just the columns we're testing in the QA sheet
        target = dfcp[target_cols]		
        result = dfcp[result_cols]

        #For writing the qa sheets, the column names need to be the same.  So we'll strip
        #  the '_qa' bit off the end of the target columns and reset the df		
        result.columns = [col[:-3] for col in result_cols]

        #for t, r in zip(target.columns, result.columns):
            #print("{:<50}{:<50}".format(t, r))
    
        #some file names	
        target_fn =  os.path.join(path, a['ddir'], '_qa_target.csv')
        result_fn =  os.path.join(path, a['ddir'], '_qa_result.csv')
        qa_fn = os.path.join(path, a['ddir'], 'agg_first_diffs.xlsx')
    
        #write the csv files
        target.to_csv(target_fn)
        result.to_csv(result_fn)
    
        #generate QA spreadsheet
    
        write_excel_qa_file(actual_fn = result_fn,
                            target_fn = target_fn,
                            index_col = firmid_col, 
                            qa_fn= qa_fn,
                            first_numeric_data_col = 1,
                            threshhold_pct_for_diffs = .01, 
                            actual_nan_values = nan_value, )
        a = 1


def fix_non_ascii(bad_list):
    """Fixes non-ascii character in a list.  Returns a list"""
    fixer = {u'“':u"'", u'”':u"'", u"’": "'", u"’": "'",}
    fixed = []
    for item in bad_list:
        if item is nan:
            fixed.append(item)
            continue
        try:
            item.decode('ascii')
            fixed.append(item)
        except:
            for junk in fixer:
                if junk in item:
                    item = item.replace(junk, fixer[junk])
            fixed.append(item)
    return fixed

def fix_non_ascii_df(df):
    """Replaces bad non-ascii characters in text (object) columns
           in a DataFrame.  Returns a DataFrame."""
    for col in df.columns:
        if df[col].dtype == object:            
            vals = list(df[col].values)
            df[col] = fix_non_ascii(vals)
    return df

def standardize_col_names(df, map_dict = {}):
    """Standardizes column names per map_dict in settings.  Solves issue of inconsistent
          nomenclature across different data products"""
    new_cols = []
    for col in df.columns:
        if col in  map_dict:
            new_cols.append(map_dict[col])
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df
      
def get_client():
    try:
        client.StopClient()
        client.Exit()
    except:
        pass
    client.StartClient()
    print("SPSS Client started.")
    return client
    
def stop_client(client):
    client.Exit()
    client.StopClient()
    print("\nSPSS Client stopped.")        
        
def abs_path(fn, path = None, ext = ''):
    """returns absolute path using settings path spec"""
    return os.path.join(os.path.abspath(path), fn) + ext
        
#These routines are needed to insure that SPSS, Excel and Python work and play
#  together vis-a-vis unicode versus ASCII single quotes
def fix_unicode_quote(strg = None):
    """replaces unicode quote with ascii one"""
    if not pd.isnull(strg):		
        return(str(strg.replace(u'\u2019', "'")))
    return strg

def clean_lookup(the_dict = None, key = None):
    """attempts to retrieve a value from a dict, fixing a unicode single quote if needed"""
    value = the_dict.get(key)
    if type(key) == str: key =  fix_unicode_quote(key)

    if not value:
        value = the_dict.get(key)
    else:
        if type(value) == str:			
            return fix_unicode_quote(value)
    return value

def extract(df = None, col_name = None, new_col_name = None, parser = None, parser_dict = None):
    """General purpose extractor.   Accepts a DataFrame and the name of the column to
       send to the parser (that's where the real action happens).  If no col_name is
       provided, it uses the index.

       Returns a DataFrame.  Column is new_column_name if provided, col_name otherwise.       
       """
    if col_name:        
        raw_names = list(df[col_name])
    else:
        raw_names = list(df.index)
        if not new_col_name: col_name = "myindex"

    names = []
    for name in raw_names:
        names.append(parser(name, parser_dict))

    #return a DataFrame; use new_col_name if provided	
    if new_col_name:
        return pd.DataFrame(data = names, columns = [new_col_name])
    return pd.DataFrame(data = names, columns = [col_name])

def get_columns_from_csv(file_name = None):
    """Starting with a .csv file name, return a list of the columns found.
       Raises FileNotFoundException with a warning if the file can't be found."""
    if not os.path.exists(file_name):
        print("Expecting, but did not find a .csv version of the last wave's .sav")
        print(file_name)
        raise FileNotFoundException        

    return list(pd.read_csv(file_name).columns)

def write_excel_qa_file(actual_fn = None,
                        target_fn = None,
                        index_col = None, 
                        qa_fn= None,
                        first_numeric_data_col = 2,
                        threshhold_pct_for_diffs = .01, 
                        actual_nan_values = ' ',
                        target_nan_values = None):
    """Writes an excel file to QA the aggregated results based on a .csv version of the
       actual data produced by a routine and the expected results.

       The QA sheet will have different tabs for target data, data produced by this routine,
       and first differences.  First difference tabs can be conditionally formatted for easy inspection.

    """
    #-- file names --
    actual = pd.read_csv(actual_fn, na_values = actual_nan_values)
    target = pd.read_csv(target_fn, na_values = target_nan_values)
    
    #set the index column if requested (default is rows)
    if index_col:
        actual.set_index(index_col, inplace = True)
        target.set_index(index_col, inplace = True)

    #-- build up first differences --

    #creates DataFrame objects; we'll keep inital left-hand columns and overwrite the numeric ones
    first_diff_abs = actual.copy()
    first_diff_pct = actual.copy()
    exceeds_threshhold = actual.copy()
    
    #new
    for df in (first_diff_abs, first_diff_pct, exceeds_threshhold):
        for col in df.columns:
            if not df[col].dtype == object:
                df[col] = nan

    #create additional Dataframes to hold the first diffs	
    for col in target.columns[first_numeric_data_col:]:
        if col == "ACSI_16_BENCH_mean":
            
            x = 1
        if not actual[col].dtype == object:        
            first_diff_abs[col] = actual[col] - target[col]
            first_diff_pct[col] = (actual[col] - target[col]) / actual[col]
            #mask =  first_diff_abs[col].mask( (abs(first_diff_abs[col] < threshhold_pct_for_diffs) ))
            mask =  first_diff_pct[col][ abs(first_diff_pct[col]) >  threshhold_pct_for_diffs]
            exceeds_threshhold[col] = mask

    #-- create the speadsheet --
    writer = pd.ExcelWriter(qa_fn)
    target.to_excel(writer, 'target', na_rep = '')
    actual.to_excel(writer, 'actual', na_rep = '')
    first_diff_pct.to_excel(writer, 'pct_diff')
    first_diff_abs.to_excel(writer, 'diff')
    exceeds_threshhold.to_excel(writer, 'diff_>_{}'.format(threshhold_pct_for_diffs))
    writer.save()

    print('File {} now available for your viewing pleasure.'.format(qa_fn)) 

def parse_excel(source = None, missing = None, test_data = None, sheetname = None):
    """Reads a spreadsheet and returns a dataframe.  This is pretty
          straightforward, generally, but is stashed here in case it
    	  needs tweaks.  Returns a DataFrame.
    """
    if test_data:
        return test_data

    df = pd.read_excel(source, sheetname = sheetname)
    return df

def drop_firms_not_in_pop_frame(data = None, df_col_name = None, using_dict = None):
    """Takes a raw DataFrame(data) and another with associated parent
    	  companies and tries to find matching parents in the vendor_id
    	  dictionary.  If parents not found, row removed.   """
    keepers = []
    good_parents = using_dict.keys()
    for ix, parent in enumerate(data[df_col_name].values):
        if clean_lookup(using_dict, parent):
            keepers.append(True)
        else:
            keepers.append(False)
    
    #set a new boolean column, use as a mask, drop, return result        
    data['keepers'] = keepers
    data = data[data.keepers]
    data = data.drop(['keepers'], axis = 1)
    return data

def read_html(data_source_name = None, header = None, skiprows = None, test_data = None):
    """Reads an html file; returns a dataframe"""
    if not test_data is None:
        return test_data
    raw, = pd.read_html(data_source_name, header = header, skiprows = skiprows)
    return raw

def get_parent_drucker_name_from_name(df = None, col_name = None, vendor_dict = None, new_col_name = None):
    """Given a DataFrame with parent names, return a DataFrame with their FirmIDs"""
    raw_names = df[df.columns[list(df.columns).index(col_name)]].copy()	
    names = []
    for name in raw_names:
        names.append(clean_lookup(vendor_dict, name)['ParentCompanyName'])
    return pd.DataFrame(names, columns = [new_col_name])

def get_parent_drucker_id_from_name(df = None, col_name = None,  new_col_name = None):
    """Given a DataFrame with parent names, return a DataFrame with their FirmIDs"""
    raw_names = df[df.columns[list(df.columns).index(col_name)]].copy()	
    ids = []
    for name in raw_names:
        ids.append(clean_lookup(py_dictionaries.parent_id, name)['firmID'])
    return pd.DataFrame(ids, columns = [new_col_name])

def get_firmid_from_vendor_name(ser, vend_dict, parent_id_dict = None, verbose = True):
    """Given a Series of vendor parent names, use the vendor dict to look up the Drucker parent name,
          then use the parent_id dict from py_dictionaries.py to return the FirmID.
          
        Returns a Series of equal length.  Names not matched  generate warnings and NaN in returned obj.
          """
    ids = []
    bad_vname = []
    bad_druck_name = []
    for vname in list(ser):
        #find the Drucker company name match to the vendor name (vendor dict)
        druck_name =  vend_dict.get(vname)
        if not druck_name:
            
            ids.append(nan)
            bad_vname.append(vname)
            continue
        
        #using the official Drucker name, from vendor dict look up the Firm ID
        co_info =  parent_id_dict.get(druck_name['ParentCompanyName'])
        if not co_info:

            ids.append (nan)
            issue = "{}".format (vname)
            bad_druck_name.append(issue)
            continue
        ids.append(co_info['firmID'])
    
    if verbose:
        
        if bad_vname:
            print("\nSorry, these names found in data, but not in vendor dict.")
            for name in bad_vname:
                print(name)
                
        if bad_druck_name:
            print("\nSorry, companies are in vendor dict, but not in the pop frame dict")
            for name in bad_druck_name:
                print(name)                
                
    return pd.Series(ids)

def fill_in_firmid_gaps(ser, vend_dict, partial_ser, verbose = True):
    """Scenario:  we have some FirmIDs associated with vendor companies that we've gotten
          by mapping these to those in the official population frame.  But the vendor
          has supplied a superset of these, along with an already-identified set of
          FirmIDs.   We'll fill in any gaps left by the original mapping (partial_ser)
          with vendor-supplied FirmIDs acquired as part of creating their vend_dict
          (provided in dictionary.xlsx).  Clarivate has this issue, for sure."""
          
    fixed = partial_ser
    
    #find any discrepencies between our vendor-provided company names and what's
    
    #create a Series out of a list of FirmIDs provided by the vendor
    fillers = pd.Series([vend_dict[name][firmid_col] for name in ser.values])
    #use these to fill in the gaps
    fixed.fillna(fillers, inplace = True)            

    return fixed
    
def validate_names(df = None, df_verbose_company_col_name = None, vendor_dict = None, excluded_dict = None,
                   parent_dict = None, source_name = None, verbose = True, df_parent_col_name = None):
    """Validates that we've accounted for every company name in the vendor data.
       This checks:
          - is each name in the vendor dict (explicity included or excluded)? and
    	  - is each associated parent company name in the Population Frame?

    	Prints out warnings if this is not the case.  Warnings suggest appropriate
    	updates.
    """
    #containers for out-of-spec names
    no_parent = []
    unaccounted = []

    parent_list = parent_dict.keys()
    if df_parent_col_name:
        df_to_validate = df_parent_col_name
    else:
        df_to_validate = df_verbose_company_col_name

    for company, verbose in zip(list(df[df_to_validate].values), list(df[df_verbose_company_col_name].values)):
        #Does the company show up in the vendor's dictionary.xls tab as a Drucker company?
        #  If so it'll have a line in the internal Python vendor dict like this:
        #  "Sears_Retailers" : {'Industry': 'Retailers', 'ParentCompanyName': u'Sears Holdings'}

        company = fix_unicode_quote(verbose)
        vendor_listing = clean_lookup(vendor_dict, verbose)
        vendor_excluded =  clean_lookup(excluded_dict, verbose)

        #is it excluded or included?
        if not vendor_listing and not vendor_excluded:
            #try for a fuzzy match
            fuzz_result = do_fuzzy_lookup(item_to_find= company, using_this= parent_list)
            if fuzz_result:
                unaccounted.append("{}|{}".format(verbose, fuzz_result))
            else:
                unaccounted.append("{}|{}".format(verbose, ""))

        #if included can it's parent be found in the Population Frame?	
        if vendor_listing:
            parent =  clean_lookup(vendor_listing, 'ParentCompanyName')
            #The drucker_listing has entries like:
                #  "Sears_Retailers" : {'Industry': 'Retailers', 'ParentCompanyName': u'Sears Holdings'}					
                #   The keys are in the form of <company>_<industry>	
            drucker_listing =  clean_lookup(parent_dict, parent)				
            if not drucker_listing:
                no_parent.append(company)
    if verbose:

        if unaccounted:
            msg =  "Warning:  these found in {}, but is not in either the included or\n "
            msg += "dictionary entries.  Please edit dictionary.xlsx and\n"
            msg += "then run build_dictionaries.py::main_create_and_screen_dicts()\n"
            msg += "to rebuild the internal Python dictionaries"
            print(msg.format(source_name))
            for name in unaccounted:
                print(name)
            print()	

        if no_parent:
            msg =  "Warning: these found in {}, I can't find a parent company.\n "
            msg += "Please edit dictionary.xlsx and then run\n"
            msg += "build_dictionaries.py::main_create_and_screen_dicts()\n"
            msg += "to rebuild the internal Python dictionaries"
            print(msg.format(source_name))
            for name in no_parent:
                print(name)

    return ({'unaccounted': unaccounted, 'no_parent': no_parent,})

def do_fuzzy_lookup(item_to_find = None, using_this = None, threshhold = 75):
    "tries to find a fuzzy match with fuzzywuzzy library"
    from fuzzywuzzy import process
    finding =  process.extractOne(item_to_find, using_this)
    matcher = finding[0]
    score = finding[1]
    if score > threshhold:
        return matcher

def DEPRECATEDmerge_spss_files(cols = None, index = None, spss_data_file_name = None):
    """Merge existing (historical) SPSS file with new one.  We'll keep all the old columns and
       add any new ones."""
    
    """Deprecated because the SPSSClient is really clunky and not terribly reliable.  New strategy is to
         simply create the .csv files for SPSS input."""
    notes = \
        """Thinking it will be easier / more straightforward to build a new spss file.   Problem is that
	           it may be better to keep the old one pristine for Larry's sanity.   So.... I need to learn how to
	           stitch SPSS files together horizontally.   This should be straightforward for ACSI due to the
	           unambiguous company names.  Might be dicier for others.   Could write vendor specific horizontal
	           stitching algorithms if push came to shove.

	           OK.  Here's how to do it:

	           test_merge1.sav  (pristine)                 test_merge_2.sav (var1 different - history revised)
	           firm     var2    var3   parent var1         firm     var2    var3   newcol1 newcol2  par var1
	           abc corp	2.00	3.00	p1	1.00           abc corp	2.00	3.00	4.00	5.00	p1	999
	           bcd corp	2.00	3.00	p1	1.00           bcd corp	2.00	3.00	6.00	7.00	p1	999
	           def corp	2.00	2.00	p2	1.00           def corp	2.00	3.00	8.00	9.00	p2	999
	                                                       new corp	333.00	444.00	555.00	666.00	p3	999
	                        var2   var3   par   var1   newcol1 newcol2   				
	            abc corp	2.00	3.00	p1	1.00	4.00	5.00
	            bcd corp	2.00	3.00	p1	1.00	6.00	7.00
	            def corp	2.00	2.00	p2	1.00	8.00	9.00
	            new corp			        p3		    555.00	666.00

	        Merging files is pretty easy.  If we do a Data..Merge..Add Variables GUI sequence, we can pick the key variable(s).
	        Both data sets need to be sorted first.  New variables are added, leaving existing ones intact.   Previous waves
	        for newly-added columns come in as missing.   Gotta to make sure to retain the existing variables and just add the new
	        columns.

	        DATASET ACTIVATE DataSet33.
	        GET FILE='C:\Users\pbarton\Desktop\drucker\current_wave_data\ACSI\test_merge_2.sav'.
	        DATASET NAME DataSet34.
	        DATASET ACTIVATE DataSet33.
	        SORT CASES BY firm parent.
	        DATASET ACTIVATE DataSet34.
	        SORT CASES BY firm parent.
	        DATASET ACTIVATE DataSet33.
	        MATCH FILES /FILE=*
	          /RENAME (newcol2 newcol1 = d0 d1) 
	          /FILE='DataSet34'
	          /RENAME (var1 var2 var3 = d2 d3 d4) 
	          /BY firm parent
	          /DROP= d0 d1 d2 d3 d4.
	        EXECUTE.

	        If, additionally, we wanted to capture the history of newly-acquire files we could merge cases (rows) to get this:

	        abc corp	2.00	3.00	p1	1.00	4.00	5.00
	        bcd corp	2.00	3.00	p1	1.00	6.00	7.00
	        def corp	2.00	2.00	p2	1.00	8.00	9.00
	        new corp			        p3		    555.00	666.00
	        abc corp	2.00	3.00	p1	999.00	4.00	5.00
	        bcd corp	2.00	3.00	p1	999.00	6.00	7.00
	        def corp	2.00	3.00	p2	999.00	8.00	9.00
	        new corp	333.00	444.00	p3	999.00	555.00	666.00

	        ADD FILES /FILE=*
	        /FILE='C:\Users\pbarton\Desktop\drucker\current_wave_data\ACSI\test_merge_2.sav'.
	      EXECUTE.

	      ... then remove the redundent (existing) cases.  Not sure how to do that algorithmically.  Not sure we need to, though
	          since the additional data represent a non-Drucker past.
	   """
def DEPRECATEDwrite_spss_aggregation_syntax(df = None,
                                  spss_aggregation_file_name = None,
                                  ratio_pair_method = None,
                                  fix_ratio_name_method = None,
                                  input_file_name = None,
                                  first_data_col_for_aggregate = None,
                                  cols_to_keep = None):
    """Deprecated because it's much more straightforward to do aggregation with
         pandas pivot table (a one-liner)"""

    """ Inputs:
    df:  A DataFrame object w/ intermediate (clean) data (used just for column names)
    spss_aggregation_file_name:  Fully-specified name of the .sav file generated
    ratio_pair_method:  Vendor-specific routine for combining benchmark/firm data
    fix_ratio_name_method:  Vendor-specific to make conformant (to 2016) ratio names
    input_file_name:  A .csv version of the intermediate (clean) data
    first_data_col_for_aggregate:  first column of real data (not firm names)
    
    Goal:  generate SPSS syntax to create an aggregated SPSS data file, combining child firms.
           - read in .csv
           - create column variable names
           - add missing value specifications
           - aggregate by (parent) FirmID
           - calculate benchmark rations firm/industry
           - calculate normalized z-scores
           - save the file as both a .sav and .csv
    """
    #set up file names, etc for use later
    if True:  #throwaway if statement - allows code folding in an IDE
             
        #a 'wave string' for labeling
        wave_as_str = str(wave)[2:]    # 2018 -> "18"	
    
        #create some fully-specified file names for spss
        base_spss_file_name = os.path.join(os.path.abspath(path), spss_aggregation_file_name) + \
            '_' + wave_as_str
        syntax_file_name =  base_spss_file_name + ".sps"
        spss_file_name = base_spss_file_name + ".sav"
        #input_file_name =  base_spss_file_name + ".csv"     #input data
        output_file_name =  base_spss_file_name + "_aggregated.csv"  #output data
    
        #last wave
        #last_base_spss_name = os.path.join(os.path.abspath(path), last_base_spss_file_name) 
        #last_spss_file_name =  last_base_spss_name + ".sav"
        #last_input_file_name =  last_base_spss_name + ".csv"
    
        #get the column names from the DataFrame (all columns)
        index = df.index.name                    #FirmID, typically
        data_column_names = list(df.columns)   
        identity_columns = data_column_names[:first_numeric_data_col]	
        paired_columns =  ratio_pair_method(df, #vendor specific
                            first_data_col_for_aggregate= first_data_col_for_aggregate) 

    
    #Write the syntax    
    syn = "\n* Syntax auto-generated from utilites.py :: write_spss_aggregation_syntax(). \n"
    #syn += "PRESERVE.\n"
    syn += "SET DECIMAL DOT. \n"

    #import the .csv
    syn += """GET DATA \n\t/TYPE=TXT \n\t/FILE= '{}'\n\t""".format(input_file_name)
    syn += """/ENCODING='UTF8'\n\t/DELIMITERS="," \n\t/QUALIFIER='"'  \n"""
    syn += """\t/ARRANGEMENT=DELIMITED \n\t/FIRSTCASE=2 \n\t/VARIABLES= \n"""

    for name in data_column_names:
        syn += "\t\t{} AUTO\n".format(fix_spss_name(name))

    syn += "\t/MAP.\n"
    #syn += "RESTORE.\n"
    #syn += "CACHE.\n"
    syn += "EXECUTE.\n"	

    #missing values	
    for name in data_column_names:
        syn += "MISSING VALUES {} ({}).\n".format (fix_spss_name(name),  nan_value)
    syn += "EXECUTE.\n"

    #aggregate    
    if True:        
        syn += "AGGREGATE\n"
        syn += "\t/OUTFILE=*  \n" 
        syn += "\t/BREAK=firmID\n"
    
        #variables loop - we're aggregating and saving the mean (skip non-data columns)
        for col in data_column_names[first_data_col_for_aggregate:]: 
            syn += "\t/{}_mean=MEAN({})\n".format(col, col)		
        syn = syn[:-1]	+ ". \n"  
        syn += "EXECUTE.\n"
    
        #get rid of duplicate firmIDs
        syn += \
            """
            * Identify Duplicate Cases.
            SORT CASES BY firmID(A).
            MATCH FILES
              /FILE=*
              /BY firmID
              /FIRST=PrimaryFirst
              /LAST=PrimaryLast.
            DO IF (PrimaryFirst).
            COMPUTE  MatchSequence1=1-PrimaryLast.
            ELSE.
            COMPUTE  MatchSequence1=MatchSequence1+1.
            END IF.
            LEAVE  MatchSequence1.
            FORMATS  MatchSequence1 (f7).
            COMPUTE  InDupGrp=MatchSequence1>0.
            MATCH FILES
              /FILE=*
              /DROP=PrimaryLast InDupGrp MatchSequence1.
            VARIABLE LABELS  PrimaryFirst 'Indicator of each first matching case as Primary'.
            VALUE LABELS  PrimaryFirst 0 'Duplicate Case' 1 'Primary Case'.
            VARIABLE LEVEL  PrimaryFirst (ORDINAL).
            EXECUTE.
    
            FILTER OFF.
            USE ALL.
            SELECT IF (PrimaryFirst=1).
            EXECUTE.
            """
        syn += "DELETE VARIABLES    PrimaryFirst.\n"

    
    #ratios
    ratio_col_names = []
    if True:        
        for firm_mean_name, ind_mean_name in paired_columns:        
            var_name =  fix_ratio_name_method(firm_mean_name) 
            ratio_col_names.append(var_name)
            syn += "COMPUTE {}={}_mean/{}_mean.\n".format(var_name, firm_mean_name, ind_mean_name)
        syn += "EXECUTE. \n"

    #do the descriptives, capturing the z-scores
    if True:        
        counter = 1
        line = []
        syn += "DESCRIPTIVES\n\t"
        syn += "VARIABLES = \n\t\t"
        #This is a bit funky - the variable names we're trying to match shifted
        #  when the "atomic" data set was produced (the original cols have _mean 
        #  appended in the AGGREGATE operation.)
        numeric_data_cols = data_column_names[first_data_col_for_aggregate:]
        for col in numeric_data_cols + ratio_col_names:
            if not col in  ratio_col_names:
                col += '_mean'
            line.append(col)
            counter += 1
            if not counter % 5:  #new row every so may variables
                syn += " ".join(line) + "\n\t\t"
                line = []
        if line: syn += " ".join(line) + "\n\t"
        syn += "/SAVE\n\t"
        syn += "/STATISTICS=MEAN STDDEV MIN MAX.\n" 
        syn += "EXECUTE.\n"    
        

        
    #Not all columns of the original data necessarily survive past this point.
    #   If provided only certain ones will be retained.   Also, this serves
    #   as a means to constrain the order of the columns (they're placed in 
    #   the same order as the columns_to_keep input).
    if cols_to_keep:
        syn += "ADD FILES FILE *\n\t"   #works on active DataSet
        syn += "KEEP\n\t"
        counter = 1
        for col in cols_to_keep:  
            line.append(col)
            counter += 1
            if not counter % 5:  #new row every so may variables
                syn += " ".join(line) + "\n\t"
                line = []
        if line:
            syn += " ".join(line) + ".\n"
        else:
            syn += ".\n"
        syn += "EXECUTE.\n"
 
    #save the SPSS file
    syn += "SAVE OUTFILE='{}'\n".format(spss_file_name)
    syn += "\tCOMPRESSED.\n"
    syn += "EXECUTE.\n"

    #save as Excel (.csv) file
    syn += "SAVE TRANSLATE OUTFILE='{}'\n".format(output_file_name)
    syn += "\t/TYPE=CSV\n"
    syn += "\t/ENCODING='UTF8'\n"
    syn += "\t/MAP\n"
    syn += "\t/FIELDNAMES\n"
    syn += "\t/CELLS=VALUES\n"
    syn += "\t/REPLACE.\n"
    syn += "EXECUTE.\n"

    #write the syntax file
    if True:        
        with open(os.path.join(path, syntax_file_name), 'w') as fh:
            fh.write(syn)
        
        print('\n')
        print("File saved as {}.".format(syntax_file_name))
        print("Run as syntax to create:")
        print("\t{}".format(spss_file_name))
        print("\t{}".format(output_file_name))
        print('\n')
    
    return {'syntax_file_name': syntax_file_name,
            'output_file_name': output_file_name,
            'syntax': syn,
           }

def fix_spss_name(name):
    """fixes a proposed SPSS name"""
    return name.replace(' ', '')


def DEPRECATEDwrite_spss_merge_syntax(old_spss_fn = None,
                            new_spss_fn = None,                            
                            old_cols = None,
                            new_cols = None, ):
    
    """Deprecated because it's much more straightforward to do merge with
         pandas; also better subject to unit-testing""" 
   
    """Merge this wave's data with last wave's, preserving the last for
       posterity."""
   
    syn = "\n\n* Merge last wave and this wave data. \n"
    #last wave
    old_dataset_name = 'pristine'
    syn += "GET FILE='{}'.\n".format(old_spss_fn)
    syn += "DATASET NAME {}.\n".format(old_dataset_name)
    syn += "SORT CASES BY {}.\n".format(spss_agg_firmID_col_name)

    #new wave
    new_dataset_name = 'new'
    
    #import the .csv
    syn += """GET DATA \n\t/TYPE=TXT \n\t/FILE= '{}'\n\t""".format(new_spss_fn)
    syn += """/ENCODING='UTF8'\n\t/DELIMITERS="," \n\t/QUALIFIER='"'  \n"""
    syn += """\t/ARRANGEMENT=DELIMITED \n\t/FIRSTCASE=2 \n\t/VARIABLES= \n"""
    for name in new_cols:
        syn += "\t\t{} AUTO\n".format(fix_spss_name(name))
    syn += ".\n"
    #syn += "GET FILE='{}'.\n".format(new_spss_fn)
    syn += "DATASET NAME {}.\n".format(new_dataset_name)		
    syn += "SORT CASES BY {}.\n".format(spss_agg_firmID_col_name)			
    syn += "DATASET ACTIVATE {}.\n".format(new_dataset_name)
    
    #merge logic
    syn += "MATCH FILES\n\t"
    syn += "/FILE=*\n\t"

    #id and rename the new columns (needed to get row-wise matching right).  The
    # list comprehension generates names d0, d1 ... dx for the new columns.
    new_cols_as_str = " ".join(new_cols)
    new_renames_as_list =  [ "d" + str(i) for i in range(len(new_cols))]
    names = " ".join(new_renames_as_list)
    #syn += "/RENAME ({} = {})\n\t".format(new_cols_as_str, names)

    #Do the same thing for the existing file, names picking up at dx+1.  Lots
    #  of extra looping code to keep line widths managable.

    # First, get rid of the FirmID and Company name (we don't want to drop those)
    for name in (spss_agg_parent_col_name, spss_agg_firmID_col_name):
        if name in old_cols: old_cols.pop(old_cols.index(name))

    syn += "/FILE='{}'\n\t".format(old_spss_fn)
    
    #RENAME  clause like /RENAME(old1 old2 = new1 new2)
    syn += "/RENAME ("
    
    #Add the old names
    counter = 1
    line = []
    for col in old_cols:
        line.append(col)
        counter += 1
        if not counter % 5:  #new row every so may variables
            syn += "   ".join(line) + "\n\t\t"
            line = []
    if line: syn += "   ".join(line) + "\n\t\t"

    #the "=" operator
    syn += "=\n\t\t"
    
    #Add the new names; list comprehension picks up where last one left off e.g., d10, d11 ...
    old_renames_as_list = [ "d" + str(i) for i in range(len(new_cols) + 1, len(new_cols) + len(old_cols) + 1)]
    
    counter = 1
    line = []
    for col in old_renames_as_list:
        line.append(col)
        counter += 1
        if not counter % 5:  #new row every so may variables
            syn += "   ".join(line) + "\n\t\t"
            line = []
    if line: syn += "   ".join(line) + "\n\t\t"
    
    #closing paren
    syn += ")\n\t"

    #BY clause - merge by FirmID and company name
    syn += "/BY {} {} \n\t".format(spss_agg_firmID_col_name, spss_agg_parent_col_name)

    #DROP clause - drop the temporary names	
    syn += "/DROP= \n\t\t"
    counter = 1
    line = []
    
    for col in old_renames_as_list:
        line.append(col)
        counter += 1
        if not counter % 15:  #new row every so may variables
            syn += "   ".join(line) + "\n\t\t"
            line = []
    if line:
        syn += "   ".join(line) + ".\n"         
    else:
        syn += ".\n"
        
    syn += "EXECUTE.\n"   #executes the merge
   
    #get rid of extaneous temp variables
    #syn += "DELETE VARIABLES {}.\n".format("  ".join(new_renames_as_list))
    
    #some fully-specified file names
    syntax_file_name = os.path.join(os.path.abspath(path), new_spss_fn[:-4] + '.sps')
    spss_file_name = os.path.join(os.path.abspath(path), new_spss_fn )
    output_file_name = os.path.join(os.path.abspath(path), new_spss_fn[:-4] + '.csv')    
    
    #save the .sav file.
    syn += "SAVE OUTFILE='{}'\n".format(spss_file_name)
    syn += "\tCOMRESSED.\n"
    syn += "EXECUTE.\n" 
    
    #save as Excel (.csv) file
    syn += "SAVE TRANSLATE OUTFILE='{}'\n".format(output_file_name)
    syn += "\t/TYPE=CSV\n"
    syn += "\t/ENCODING='UTF8'\n"
    syn += "\t/MAP\n"
    syn += "\t/FIELDNAMES\n"
    syn += "\t/CELLS=VALUES\n"
    syn += "\t/REPLACE.\n"
    syn += "EXECUTE.\n" 

    #write the syntax file
    with open(syntax_file_name, 'w') as fh:
        fh.write(syn)
    print("Syntax file saved as {}.".format(syntax_file_name))
    
    print('')
    print("Created {}.\nRun as syntax to produce:\n\t{}.".format(syntax_file_name, spss_file_name))
    print("\tand {}.".format(output_file_name))
    print('')    

    return {'spss_file_name': spss_file_name, 
            'syntax_file_name': syntax_file_name,
            'syntax': syn,
            'output_file_name': output_file_name,
            }
    
    
def scan_log_files(lst_of_logs):    
    """Does a first-order scan of SPSS error logs to flag errors.
         Input is an iterable of file names.   Does nothing if the
         file does not exist.  Returns a list of output."""
    output = []
    for log in lst_of_logs:
        if os.path.exists(log):        
            with open(log) as fh:
                rows=csv.reader(fh)
                for row in rows:
                    if row:                
                        if "Error" in row[0] or "Warning" in row[0]:
                            output.append("Warning:  Error(s) or Warning(s) found in {}".format(log))
                            output.append("")
                            break

def aggregate(df, index_col = 'FirmID', drop_for_agg = ['VendorID', 'VendorName']):
    "Aggregate across companies to get mean values"
    try:
        nan_value
    except:
        nan_value = '99999'
    
    df = df.replace(nan_value, nan)
    #aggregate the numeric columns, preserving their names
    agg = pd.pivot_table(df, index = index_col)
    df.set_index(index_col, inplace = True)
    #replace numeric cols in original w/ newly-aggregated ones
    for col in agg.columns:
        df[col] = agg[col]

    #drop unneeded columns
    df.drop(drop_for_agg, axis = 1, inplace = True)

    #dedup rows
    df.drop_duplicates(inplace = True)

    df.reset_index(inplace = True)

    return df


def pad(df = None, waves_to_update = 1, vendor = 'ACSI', wave = 2016):
    """Adds replaces padded columns and updates w/ new wave data"""
    try:
        nan_value
    except:
        nan_value = 99999
    
    df = df.replace(nan_value, nan)

    #Add a PADDED column for this wave
    #whistle up the names of the columns to update here, then update with new data
    update_cols = []
    first_yr_to_update = wave - waves_to_update + 1

    #update the PADDED columns    
    padded_cols = []

    #company data
    base_name = vendor + '_' + str(wave)[2:]
    new_col_name =  base_name + '_PADDED'
    df.insert(df.columns.get_loc(base_name) + 1, new_col_name, NaN)
    padded_cols.append(new_col_name)

    #benchmark data
    base_name =  'ACSI_' + str(wave)[2:] + "_BENCH"
    new_col_name =  base_name + '_PADDED'
    df.insert(df.columns.get_loc(base_name) + 1,new_col_name, NaN)
    padded_cols.append(new_col_name)        

    #fill in the padded col values
    for col_name in padded_cols:
        #find the indices for the last two waves
        this_col_ix = df.columns.get_loc(col_name)        
        current_data_col_ix = this_col_ix - 1
        last_data_col_ix = current_data_col_ix - 1

        #find their names
        current_data_col_name = df.columns[current_data_col_ix]
        last_data_col_name = df.columns[last_data_col_ix]

        #give the padded cell this year's value
        df[col_name] = df[current_data_col_name]
        #get a mask for the n/a rows
        null_rows =  pd.isnull(df[col_name])
        #give them last year's data

        #  tempting, but has nested getters 
        #       df[col_name][null_rows] =df[last_data_col_name]
        df.loc[null_rows, col_name ] = df.loc[null_rows, last_data_col_name]
        x = 1
    return df


def merge(old = None, new = None, waves_to_update = 2, last_wave = 2016,
          replication_mode = False, retain_old_padded = False):
    """Merges exsting data with new, keeping old intact.  Some wrinkles here:
        - overwrite any 'PADDED' / missing data from previous years with newly-available
    	  if replication_mode =False; otherwise leave previsously used data intact for posterity.
    	- add waves_to_update new columns.  Could mean a "gap year" if waves not consecutive.

    	Works with .csv versions of both the new and old data.  Finds insertion point by the last two
    	digits of last_wave."""
    try:
        wave
    except:
        wave = 2018

    #Do we have padded data?  If so make a list of them.
    padded_cols = []
    for ix, col in enumerate(old.columns):
        if "PADDED" in col:
            padded_cols.append(ix) 
    if padded_cols:
        if replication_mode:
            #replace incomplete last year's data with the padded version (what we last reported)
            for col in padded_cols:				
                old.iloc[:, col-1] = old.iloc[:, col]

        if not retain_old_padded:
            old.drop(old.columns[padded_cols], axis = 1, inplace = True)

    #Figure out which columns to add then make it happen.	

    #List comprehension produces for wave=2018, waves_to_update=2:  ['17', '18'].  For
    #  wave=2018, waves_to_update=1: ['18'].  This will flag the cols we want to add.
    add_me_flags = [str(i)[2:] for i in range(wave - waves_to_update + 1, wave + 1)]

    #Look for the 'flag' in the column name.  If found, add to empty list.  cols_to_add
    #  will be something like ['ACSI_18', 'ACSI_18_BENCH'].  Key thing is that the two-digit
    #  year is the indicator.
    cols_to_add = []
    for col in new.columns:
        for flag in add_me_flags:
            if flag in col:
                cols_to_add.append(col)

    ##This assumes that both have a column 'FirmID'.  Should make this a setting.
    #Merge the new columns, aligning data on FirmID
    new.reset_index(inplace = True, drop = True)
    new.set_index('FirmID', inplace = True)
    old.reset_index(inplace = True, drop = True)
    old.set_index('FirmID', inplace = True)	


    for col in cols_to_add:
        old[col] = new[col]

    return old	 #really, an augmented version



def verify_drucker_names(fn, namecol = 'CompanyName', firmidcol = 'FirmID'):
    """Verify that Drucker names are correct in the data file.  If not, report and
          fix.	  
    	  """

    #drucker_col = [raw_col_map]["Drucker Name"]
    #firmid_col = [raw_col_map]["FirmID"]

    df = pd.read_csv(fn)
    df.set_index(firmid_col, inplace = True)

    #make a test df with the CSRHub Drucker name and what's in the official dict

    test = df[drucker_co].copy()
    test['id'] = test[firmid_col]
    test['target'] = id_parent[test[firmid_col]]	
    test['mask'] = test[drucker_col] == test['target']
    wrong = df.query('mask == 0')
    if wrong.any():		
        print(wrong[:])
        input("continue? ")
        
def check_for_new_company(sdf = None, sdf_fn = None,
                          ddf = None, ddf_fn = None, verbose = True):
    """  Check to see if the vendor has a FirmID that is not in its "destination" table
             but is in the current pop frame.  Returns rows to be added to destination.
                
        NB:  this only checks the FirmID field.  Use check_company_firmid() to check
        company name and association with right ID against the popframe
        
        Its purpose is to identify rows that need to be added to the destination.
        """
    
    #check the row index (FirmID); returns a DataFrame with firms in vendor but not
    #  in destination.
    new_rows = sdf[~sdf.isin(ddf)].dropna(how = 'all')
    
    #if we get anything, check if it's in the population frame
    if len(new_rows):
        ids_in_popfr = []
        ids = list(new_rows.index)
        for i in ids:
            #if we find it in the popfr, remember it
            if i in id_parent:
                ids_in_popfr.append(i)
    
    #screen out firms not in popfr; we'll return this            
    new_and_in_pop_frame = new_rows.drop(ids_in_popfr)
    
    if verbose and len(new_and_in_pop_frame):
        print("Found these rows in vendor data, but not in target")
        print(new_and_in_pop_frame)
        
    a = 1
        
    return  new_and_in_pop_frame




def check_company_firmid(sdf, verbose = False):
    """Checks whether the company / firmid pairings in vendor data are still good.
         Potential issues:
            - a buyout has reassigned a company to a different parent;
            - the company name is mismatched to the Drucker version.
            
        Returns a DataFrame of companies in the popfr, but with mismatched names (providing
           the input has a company name field) , None otherwise
            """
    
    firmid_not_in_pf =  [firmid for firmid in list(sdf.index) if not firmid in id_parent]        
    firmid_in_pf =  [firmid for firmid in list(sdf.index) if firmid in id_parent]

    #Check the company name. This is worse than need be during the transition because
    #  the column is inconsistently named.   Worse, some data tables e.g., the employee
    #  committment rollup don't have company names.  This is crude, but we'll look for some
    #  permutations in the DataTable columns.  If we find one, we'll check the name against the
    #  popframe dict.
    
    #find the company column
    co_name_cols = [col for col in sdf.columns if col.lower() in ['company name', 'companyname']]
    
    #if we've found one, proceed
    if len(co_name_cols) == 1:
        col = co_name_cols[0]
        
        #screen out companies we won't find in the popfr
        good_sdf = sdf.copy().drop(firmid_not_in_pf)
        good_sdf.sort_index(inplace = True)
        
        #create a fake column for the index
        good_sdf['xid'] = good_sdf.index
        
        #get the official firm names from the dict and add as a column
        firms_from_pf = [id_parent[fid]['company'] for fid in list(good_sdf['xid'])]
        good_sdf['dict'+ col] = firms_from_pf
        
        #find mismatches
        mismatches = good_sdf[good_sdf[col] !=  good_sdf['dict'+ col]]
        
        if verbose:
            print("These company names do not match the Population Frame's version")
            print(mismatches)
        
        return mismatches
    
    
def test_outliers(df = None, threshhold = 3, verbose = False, first_data_col = 1, name_or_fn = None):
    """Takes a dataframe as input.   Creates column-wise z-scores.
          Checks for outliers beyond |threshhold|.  Returns a
          dataframe containing only the rows w/ bad data, and only
          the bad values."""
    
    df = df.iloc[:, first_data_col:]
    #create the z-scores and screen for rows w/ elements out-of-whack
    zdf = compute_z(df, first_data_col = 0)
    zcols = zdf.columns
    
    #which ones are wonky?
    bad_as_tf = abs(zdf) > threshhold
    bad_as_nan = bad_as_tf.replace(False, nan)
    bad_as_nan.dropna(how = 'all', inplace = True)

    #we're interested in the out-of-spec rows
    only_bad_rows = df.iloc[ df.index.isin(bad_as_nan.index), :]
    
    bad_as_nan.columns = only_bad_rows.columns
    out = bad_as_nan * only_bad_rows

    if verbose:
        print('Data in {} has outliers'.format(name_or_fn))
        print(out)

    return out

def aggregate_append(df = None, index_col = None, rollup_suffix = '_mean', first_data_col = None):
    """Input dataframe and a break column.  Returns a dataframe w/new column, containing
         aggregated averages.   For instance, if we needed to roll industry averages and
        assign the average as a new column, this is the go-to routine."""
    
        
    #make pivot table on break variable (default agg method is mean())
    agg = pd.pivot_table(df, index = index_col)
    

    #The agg function is already row-indexed by the break variable
    #   so reindex the original df to the same
    
    #df.reset_index(inplace = True)
    df.insert(0, firmid_col, df.index)
    #df[df.index.name] = df.index
    
    df.set_index(index_col, inplace = True)
    
    #create new columns (named for originals) to hold the averaged data

    for name in list(df.columns[first_data_col:]):
        df[str(name) + rollup_suffix] = agg[name]        

    df.set_index(firmid_col, inplace = True)
    
    return df

def test_dupes(df, name_or_fn = None, verbose = False):
    dupes = df[df.duplicated]
    if len(dupes):
        print('Source data {} has {} duplicate rows'.format(name_or_fn, len(dupes)))
        print('First is:\n'.format(dupes[0]))     


def check_bad_names_in_dict(names_list = None, mydict = None):
    """Takes a list-like object with names and the dict they should occupy as keys.
           Raises NameNotFoundError if any missing."""        
    bad = []
    for name in names_list:
        if not name in mydict:
            bad.append(name)
        
    if bad:
        for b in bad:
            print(b)
        raise NameNotFoundError(bad)