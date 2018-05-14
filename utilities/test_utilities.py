#test_utilities.py
# -*- coding: utf-8 -*-
"""Test various utility functions used with multiple vendors"""

from __future__ import print_function
import unittest

import pandas as pd
import os
import sys
from numpy import nan
import csv
import SpssClient as client
from SpssClient import  SpssClientException

from utilities import aggregate, pad, merge, fix_non_ascii, compute_z, poke_in_col, \
     average_cols, check_for_new_company, check_company_firmid, test_outliers, \
     get_firmid_from_vendor_name, aggregate_append, fill_in_firmid_gaps
#supresses warnings (complains when attempting to round NaN)
import warnings
warnings.filterwarnings('ignore')

from general_settings.general_settings import *

#from utilities import abs_path

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)
pd.set_option('max_rows', 1000)

waves_to_update = 2  #including this one!
last_wave = 2016
replication_mode = False

def gen_data(fn, rows_keep = 10, init_cols = 2, last_data_cols = 5, 
             saveas = 'scratch.csv', verbose = False):
    """Allows you to specify which rows/cols of a csv file you want to use for
          a test file.  (First couple of cols, and ones at end; a few rows typically).
    	  saves it as a csv and prints it to console.  Fast way to build up testing
    	  assets."""
    vendor = 'ACSI'
    path = os.path.join("..", "..", "current_wave_data")
    full_path = os.path.join(os.path.abspath(path), vendor, fn)
    csv = pd.read_csv(full_path)
    csv.replace(nan_value, nan, inplace = True)

    good_cols = range(init_cols) + [col for col in range(-last_data_cols, 0)]
    keep = csv.iloc[:rows_keep, good_cols]

    save_path = os.path.join(os.path.abspath(path), vendor, saveas)


    if verbose:			
        print(save_path)
        print("(from {})".format(full_path))
        print()
        print(keep.__repr__())
        print()

    #keep.to_csv(save_path, na_rep = nan, index = False)

    return(keep)





def write_scratch_qa_files_for_augmentation():
    """Rrite scratch qa files - likely to be edited.  Run once to generate csv files.
       Edit in excel to prune.  """
    verbose = True

    #fn = 'Atomic N=732 With Just ACSI.csv'
    #saveas =  'test_atomic_16.csv'
    #gen_data(fn, rows_keep = 3, init_cols= 4, last_data_cols = 50, saveas = saveas, verbose = verbose)	

    fn = "ACSI Clean and Augmented.csv"
    saveas =  'test_data_acsi_clean_16.csv'
    gen_data(fn, rows_keep = 9, init_cols= 4, last_data_cols = 4, saveas = saveas, verbose = verbose)


    fn = "ACSI_Results_2018.csv"
    #fn = "ACSI_Clean_and_Augmented_qa_18.csv"
    saveas =  'test_data_acsi_augmented_16.csv'
    gen_data(fn, rows_keep = 9, init_cols= 4, last_data_cols = 5, saveas = saveas, verbose = verbose)

    before = """"""
    after = """"""

#write_scratch_qa_files_for_augmentation()

class TestPokeInCol(unittest.TestCase):
    """Test poke_in_a_col() function."""
    
    """Specs include:  source DataFrame  sdf
                       source cols        [scol]
                       destination DataFrame ddf
                       destination cols *after which* to insert the new column [dcol]
                       qa_mode - adds a "_qa" to the destination column name

        The source and destination DataFrames should already be indexed correctly

    General-purpose method for updating data tables.    
    """      
    
        
    def test_poke_col(self):      
        
        #-- test a single column --
        source_data = [['firm1', 1, 2, 3],
                       ['firm2', 10, 20, 30],
                       ['firm3', 100, 200, 300],
                       ['firm4', 100, 200, 300]  #not in destination
                       ]
        source_cols = ['firm', 's1', 's2', 's3']
        
        dest_data = [['firm0', 5, 6, 7, 8, 9],         #not in source
                     ['firm2', 50, 60, 70, 80, 90],
                     ['firm3', 500, 600, 700, 800, 900],
                     ['firm4', 1500, 1600, 1700, 1800, 1900]
                     ]
        dest_cols = ['firm', 'd1', 'd2', 'd3', 'd4', 'd5']

        scol = 's1'
        dcol = 'd1'
        index = 'firm'
        sdf = pd.DataFrame(source_data, columns = source_cols)
        sdf.set_index(index, inplace = True)
        ddf = pd.DataFrame(dest_data,  columns = dest_cols)
        ddf.set_index(index, inplace = True)
        
        returned = poke_in_col(sdf = sdf, scol = scol, ddf = ddf, dcol = dcol)
        ##todo:  check result index
        
        expected_data = [[5, nan, 6, 7, 8, 9],         #not in source
                         [50, 10, 60, 70, 80, 90],
                         [500, 100, 600, 700, 800, 900],
                         [1500,200, 1600, 700, 1800, 1900]
                         ]        
        
        expected_cols = ['d1', scol, 'd2', 'd3', 'd4', 'd5']
        expected = pd.DataFrame(expected_data,  columns = expected_cols)
        #gotta to exchange the nans for rounding
        returned.replace(nan, nan_value, inplace = True)
        expected.replace(nan, nan_value, inplace = True)

        for col, dtype in zip(expected_cols, expected.dtypes):
            if not dtype == object:
                diff = expected[col].round(3) - returned[col].round(3)
                self.assertFalse(diff.any())

        returned.replace(nan_value, nan, inplace = True)
        expected.replace(nan_value, nan, inplace = True)

        self.assertEqual(list(returned.columns ), list(expected.columns))
        a = 1
         #-- test multiple cols --
    def test_poke_multiple_col(self):
        
        #-- test a single column --
        source_data = [['firm1', 1, 2, 3],
                       ['firm2', 10, 20, 30],
                       ['firm3', 100, 200, 300],
                       ['firm4', 100, 200, 300]  #not in destination
                       ]
        source_cols = ['firm', 's1', 's2', 's3']
        
        dest_data = [['firm0', 5, 6, 7, 8, 9],         #not in source
                     ['firm2', 50, 60, 70, 80, 90],
                     ['firm3', 500, 600, 700, 800, 900],
                     ['firm4', 1500, 1600, 1700, 1800, 1900]
                     ]
        dest_cols = ['firm', 'd1', 'd2', 'd3', 'd4', 'd5']

        scol = ['s1', 's2', 's3']
        dcol = ['d1', 'd2', 'd3']
        index = 'firm'
        sdf = pd.DataFrame(source_data, columns = source_cols)
        sdf.set_index(index, inplace = True)
        ddf = pd.DataFrame(dest_data,  columns = dest_cols)
        ddf.set_index(index, inplace = True)
        
        returned = poke_in_col(sdf = sdf, scol = scol, ddf = ddf, dcol = dcol)
        ##todo:  check result index
        
        expected_data = [[5, nan, 6, nan, 7, nan, 8, 9],         #not in source
                         [50, 10, 60, 20, 70, 30, 80, 90],
                         [500, 100, 600, 200, 700, 300, 800, 900],
                         [1500, 200, 1600, 200, 700, 300, 1800, 1900]
                         ]        
        
        expected_cols = ['d1', scol[0], 'd2', scol[1], 'd3', scol[2], 'd4', 'd5']
        expected = pd.DataFrame(expected_data,  columns = expected_cols)
        #gotta to exchange the nans for rounding
        returned.replace(nan, nan_value, inplace = True)
        expected.replace(nan, nan_value, inplace = True)

        for col, dtype in zip(expected_cols, expected.dtypes):
            if not dtype == object:
                diff = expected[col].round(3) - returned[col].round(3)
                self.assertFalse(diff.any())

        returned.replace(nan_value, nan, inplace = True)
        expected.replace(nan_value, nan, inplace = True)

        self.assertEqual(list(returned.columns ), list(expected.columns))         

class TestComputeZ(unittest.TestCase):

    def test_return_z(self):
        """Can we return just z-scores?"""

        data = [['text', 25, 10 ],
                ['text',60, 20],
                ['text',80, 30], 
                ['text',90, nan]
                ]
        data_cols = ['text','col0', 'col1']

        expected_data = [['text',-1.35081, -1],
                         ['text', -.13072, 0],
                         ['text',.56647, 1], 
                         ['text',.91506, nan]
                         ]
        expected_cols = ['text','Zcol0', 'Zcol1']

        inp =pd.DataFrame(data = data, columns = data_cols)

        returned = compute_z(inp, first_data_col= 1)		
        expected = pd.DataFrame(data = expected_data, columns = expected_cols)

        #gotta to exchange the nans for rounding
        returned.replace(nan, nan_value, inplace = True)
        expected.replace(nan, nan_value, inplace = True)

        for col, dtype in zip(expected_cols, expected.dtypes):
            if not dtype == object:
                diff = expected[col].round(3) - returned[col].round(3)
                self.assertFalse(diff.any())

        returned.replace(nan_value, nan, inplace = True)
        expected.replace(nan_value, nan, inplace = True)

        self.assertEqual(list(returned.columns ), list(expected.columns))

    def test_return_z_appended(self):
        """Can we return appended z-scores?"""

        data = [['text', 25, 10 ],
                ['text',60, 20],
                ['text',80, 30], 
                ['text',90, nan]
                ]
        data_cols = ['text','col0', 'col1']

        expected_data = [['text', 25, 10,-1.35081, -1],
                         ['text',60, 20, -.13072, 0],
                         ['text',80, 30,.56647, 1], 
                         ['text',90, nan,.91506, nan]
                         ]
        expected_cols = ['text','col0', 'col1', 'Zcol0', 'Zcol1']

        inp =pd.DataFrame(data = data, columns = data_cols)

        returned = compute_z(inp, first_data_col= 1, append_cols = True)		
        expected = pd.DataFrame(data = expected_data, columns = expected_cols)

        #gotta to exchange the nans for rounding
        returned.replace(nan, nan_value, inplace = True)
        expected.replace(nan, nan_value, inplace = True)

        for col, dtype in zip(expected_cols, expected.dtypes):
            if not dtype == object:
                diff = expected[col].round(3) - returned[col].round(3)
                self.assertFalse(diff.any())

        returned.replace(nan_value, nan, inplace = True)
        expected.replace(nan_value, nan, inplace = True)

        self.assertEqual(list(returned.columns ), list(expected.columns))

    def test_return_t_appended(self):
        """Can we return appended z-scores?"""

        data = [['text', 25, 10 ],
                ['text',60, 20],
                ['text',80, 30], 
                ['text',90, nan]
                ]
        data_cols = ['text','col0', 'col1']

        expected_data = [['text', 25, 10, 36.491915, 40],
                         ['text',60, 20, 48.692766, 50],
                         ['text',80, 30, 55.664681, 60], 
                         ['text',90, nan, 59.150638, nan]
                         ]
        expected_cols = ['text','col0', 'col1', 'Tcol0', 'Tcol1']

        inp =pd.DataFrame(data = data, columns = data_cols)

        returned = compute_z(inp, first_data_col= 1, append_cols = True, as_tscore = True)		
        expected = pd.DataFrame(data = expected_data, columns = expected_cols)

        #gotta to exchange the nans for rounding
        returned.replace(nan, nan_value, inplace = True)
        expected.replace(nan, nan_value, inplace = True)

        for col, dtype in zip(expected_cols, expected.dtypes):
            if not dtype == object:
                diff = expected[col].round(3) - returned[col].round(3)
                self.assertFalse(diff.any())

        returned.replace(nan_value, nan, inplace = True)
        expected.replace(nan_value, nan, inplace = True)

        self.assertEqual(list(returned.columns ), list(expected.columns))		



class TestAsciiRepair(unittest.TestCase):

    def test_nonascii(self):
        "tests removal of select non-ascii characters"
        to_test = [ (u'Amazon.com, Inc.', u'Amazon.com, Inc.'),
                    (u'Land O’Lakes', u"Land O'Lakes"),
                    (u'Toys “R” Us', u"Toys 'R' Us")
                    ]
        bad =[original for original, target in to_test]
        good = [target for original, target in to_test]
        result = fix_non_ascii(bad)
        self.assertEqual(result, good)



class TestMerge(unittest.TestCase):

    def setUp(self):
        """datasets defined here"""

        self.vendor = "test_vendor"
        #this year's fresh vendor data	
        incoming_fn = 'test_merge_new_data.csv'
        self.incoming = pd.read_csv(os.path.join(os.path.abspath(path), self.vendor, incoming_fn))

        #existing data (padded)
        old_padded_fn = 'test_merge_old_pad.csv'
        self.oldpad = pd.read_csv(os.path.join(os.path.abspath(path), self.vendor, old_padded_fn))

        #existing data (not padded)
        old_nopad_fn = 'test_merge_old_nopad.csv'
        self.oldnopad = pd.read_csv(os.path.join(os.path.abspath(path), self.vendor, old_nopad_fn))


        self.last_wave = 2016

    def test_padded_replicate_no_retain_1_wave(self):

        """replicate old data (e.g., keep 2016 padded as official 2016)
		      no-retain PADDED columns from last wave
			  add 1 wave of data"""
        target_data = \
            """
o
   FirmID              CompanyName                                         VendorName  ACSI_15  ACSI_16  ACSI_16_PADDED  ACSI_15_BENCH  ACSI_16_BENCH  ACSI_16_BENCH_PADDED
0       1               Apple Inc.                        Apple [Cellular Telephones]       80     81.0              81             79           79.0                    79
1       1               Apple Inc.                         Apple [Personal Computers]       84      NaN              84             78            NaN                    78
2       2           Alphabet, Inc.   Google [Internet Search Engines and Information]       78      NaN              84             77            NaN                    77
3       2           Alphabet, Inc.                    Google+ [Internet Social Media]       75     76.0              76             73           73.0                    73
4       2           Alphabet, Inc.           YouTube (Google) [Internet Social Media]       76     77.0              77             73           73.0                    73
5       3    Microsoft Corporation  Bing (Microsoft) [Internet Search Engines and ...       72     75.0              75             77           77.0                    76
6       3    Microsoft Corporation                      Microsoft [Computer Software]       75     80.0              80             81           81.0                    78
7       3    Microsoft Corporation                                         VendorName       75     74.0              74             79           79.0                    79
8       3    Microsoft Corporation                        Apple [Cellular Telephones]       74     75.0              75             77           77.0                    76
9       4  Exxon Mobil Corporation                         Apple [Personal Computers]    99999  99999.0           99999          99999        99999.0                    76
t
   FirmID              CompanyName                                         VendorName  ACSI_15  ACSI_16  ACSI_18  ACSI_15_BENCH  ACSI_16_BENCH  ACSI_18_BENCH
0       1               Apple Inc.                        Apple [Cellular Telephones]       80       81     81.0             79             79             81
1       1               Apple Inc.                         Apple [Personal Computers]       84       84      NaN             78             78             82
2       2           Alphabet, Inc.   Google [Internet Search Engines and Information]       78       84      NaN             77             77             83
3       2           Alphabet, Inc.                    Google+ [Internet Social Media]       75       76     81.0             73             73             84
4       2           Alphabet, Inc.           YouTube (Google) [Internet Social Media]       76       77     74.0             73             73             85
5       3    Microsoft Corporation  Bing (Microsoft) [Internet Search Engines and ...       72       75     73.0             77             76             86
6       3    Microsoft Corporation                      Microsoft [Computer Software]       75       80     76.0             81             78             87
7       3    Microsoft Corporation                                         VendorName       75       74     80.0             79             79             88
8       3    Microsoft Corporation                        Apple [Cellular Telephones]       74       75     72.0             77             76             89
9       4  Exxon Mobil Corporation                         Apple [Personal Computers]    99999    99999  99999.0          99999             76             90


		    """		    
        new_data = self.incoming
        old_data = self.oldpad

        target_fn = 'test_merge_pad_replicate_noretain_1_wave.csv'
        target = pd.read_csv(os.path.join(os.path.abspath(path), self.vendor, target_fn))

        waves_to_update = 1
        last_wave = self.last_wave
        replication_mode = True
        retain_old_padded= False

        n = new_data; o = old_data; t = target

        result = merge(old = old_data, new = new_data,
                       waves_to_update = waves_to_update, last_wave =self.last_wave,
                       replication_mode = replication_mode, retain_old_padded= retain_old_padded)
        r = result

        target.reset_index(inplace = True, drop = True)
        result.reset_index(inplace = True)

        target.sort_index(axis = 1, inplace = True)
        result.sort_index(axis = 1, inplace = True)

        #make sure all numeric data match up
        for col, dtype in zip(result.columns, result.dtypes):			
            if not dtype == object:
                diff = (result[col] - target[col]).round(5)
                self.assertFalse(diff.any())

        #make sure all columns exist
        diff = result.columns.sort_values() == target.columns.sort_values()
        self.assertTrue(diff.all())
        a = 1


    def test_padded_replicate_no_retain_3_wave(self):

        """replicate old data (e.g., keep 2016 padded as official 2016)
		      no-retain PADDED columns from last wave
			  add 3 waves of data"""
        target_data = \
            """
o
   FirmID              CompanyName                                         VendorName  ACSI_15  ACSI_16  ACSI_16_PADDED  ACSI_15_BENCH  ACSI_16_BENCH  ACSI_16_BENCH_PADDED
0       1               Apple Inc.                        Apple [Cellular Telephones]       80     81.0              81             79           79.0                    79
1       1               Apple Inc.                         Apple [Personal Computers]       84      NaN              84             78            NaN                    78
2       2           Alphabet, Inc.   Google [Internet Search Engines and Information]       78      NaN              84             77            NaN                    77
3       2           Alphabet, Inc.                    Google+ [Internet Social Media]       75     76.0              76             73           73.0                    73
4       2           Alphabet, Inc.           YouTube (Google) [Internet Social Media]       76     77.0              77             73           73.0                    73
5       3    Microsoft Corporation  Bing (Microsoft) [Internet Search Engines and ...       72     75.0              75             77           77.0                    76
6       3    Microsoft Corporation                      Microsoft [Computer Software]       75     80.0              80             81           81.0                    78
7       3    Microsoft Corporation                                         VendorName       75     74.0              74             79           79.0                    79
8       3    Microsoft Corporation                        Apple [Cellular Telephones]       74     75.0              75             77           77.0                    76
9       4  Exxon Mobil Corporation                         Apple [Personal Computers]    99999  99999.0           99999          99999        99999.0                    76

t
   FirmID              CompanyName                                         VendorName  ACSI_15  ACSI_16  ACSI_17  ACSI_18  ACSI_15_BENCH  ACSI_16_BENCH  ACSI_17_BENCH  ACSI_18_BENCH
0       1               Apple Inc.                        Apple [Cellular Telephones]       80       81       81     81.0             79             79             79             81
1       1               Apple Inc.                         Apple [Personal Computers]       84       84       83      NaN             78             78             77             82
2       2           Alphabet, Inc.   Google [Internet Search Engines and Information]       78       84       82      NaN             77             77             76             83
3       2           Alphabet, Inc.                    Google+ [Internet Social Media]       75       76       81     81.0             73             73             73             84
4       2           Alphabet, Inc.           YouTube (Google) [Internet Social Media]       76       77       74     74.0             73             73             73             85
5       3    Microsoft Corporation  Bing (Microsoft) [Internet Search Engines and ...       72       75       73     73.0             77             77             76             86
6       3    Microsoft Corporation                      Microsoft [Computer Software]       75       80       76     76.0             81             81             78             87
7       3    Microsoft Corporation                                         VendorName       75       74       80     80.0             79             79             79             88
8       3    Microsoft Corporation                        Apple [Cellular Telephones]       74       75       72     72.0             77             77             76             89
9       4  Exxon Mobil Corporation                         Apple [Personal Computers]    99999    99999    99999  99999.0          99999          99999             76             90


		    """		    
        new_data = self.incoming
        old_data = self.oldpad

        target_fn = 'test_merge_pad_replicate_noretain_3_wave.csv'
        target = pd.read_csv(os.path.join(os.path.abspath(path), self.vendor, target_fn))

        waves_to_update = 3
        last_wave = self.last_wave
        replication_mode = True
        retain_old_padded = False

        n = new_data; o = old_data; t = target

        result = merge(old = old_data, new = new_data,
                       waves_to_update = waves_to_update, last_wave =self.last_wave,
                       replication_mode = replication_mode, retain_old_padded= retain_old_padded)
        r = result

        target.reset_index(inplace = True, drop = True)
        result.reset_index(inplace = True)

        target.sort_index(axis = 1, inplace = True)
        result.sort_index(axis = 1, inplace = True)

        #make sure all numeric data match up
        for col, dtype in zip(result.columns, result.dtypes):			
            if not dtype == object:
                diff = (result[col] - target[col]).round(5)
                self.assertFalse(diff.any())

        #make sure all columns exist
        diff = result.columns.sort_values() == target.columns.sort_values()
        self.assertTrue(diff.all())


        a = 1




class TestPad(unittest.TestCase):
    """Tests padding operation"""

    def setUp(self):
        pass
    def test_one_wave(self):
        test_data = \
            """
		    oo
		    FirmID                                         VendorName  ACSI_14  ACSI_15  ACSI_16  ACSI_14_BENCH  ACSI_15_BENCH  ACSI_16_BENCH
		 0       1                        Apple [Cellular Telephones]       79     80.0     81.0             78           78.0           79.0
		 1       1                         Apple [Personal Computers]       84     84.0      NaN             78           77.0            NaN
		 2       2   Google [Internet Search Engines and Information]       83     78.0      NaN             80           76.0            NaN
		 3       2                    Google+ [Internet Social Media]       84      NaN      NaN             71            NaN            NaN
		 4       2           YouTube (Google) [Internet Social Media]       78     76.0     77.0             71           74.0           73.0
		 5       3  Bing (Microsoft) [Internet Search Engines and ...       73     72.0     75.0             80           76.0           77.0
		 6       3                      Microsoft [Computer Software]       75     75.0     80.0             76           74.0           81.0

		 t
		    FirmID                                         VendorName  ACSI_14  ACSI_15  ACSI_16  ACSI_16_PADDED  ACSI_14_BENCH  ACSI_15_BENCH  ACSI_16_BENCH  ACSI_16_BENCH_PADDED
		 0       1                        Apple [Cellular Telephones]       79     80.0     81.0            81.0             78           78.0           79.0                  79.0
		 1       1                         Apple [Personal Computers]       84     84.0      NaN            84.0             78           77.0            NaN                  77.0
		 2       2   Google [Internet Search Engines and Information]       83     78.0      NaN            78.0             80           76.0            NaN                  76.0
		 3       2                    Google+ [Internet Social Media]       84      NaN      NaN             NaN             71            NaN            NaN                   NaN
		 4       2           YouTube (Google) [Internet Social Media]       78     76.0     77.0            77.0             71           74.0           73.0                  73.0
		 5       3  Bing (Microsoft) [Internet Search Engines and ...       73     72.0     75.0            75.0             80           76.0           77.0                  77.0
		 6       3                      Microsoft [Computer Software]       75     75.0     80.0            80.0             76           74.0           81.0                  81.0


		    """		    
        ofn = 'test_pad_original_from_ACSI_clean_and_augmented.csv' 
        vendor = 'test_vendor'
        self.orig = pd.read_csv(os.path.join(os.path.abspath(path), vendor, ofn))

        tfn = 'test_pad_target_ACSI_clean_and_augmented_1_wave.csv'
        self.target = pd.read_csv(os.path.join(os.path.abspath(path), vendor, tfn))

        t = self.target
        o = self.orig		

        result = pad(self.orig, waves_to_update = 1, vendor = 'ACSI', wave = 2016)

        r = result

        for col, dtype in zip(result.columns, result.dtypes):			
            if not dtype == object:
                diff = result[col].round(5) - self.target[col].round(5)
                self.assertFalse(diff.any())
        a = 1

    def test_two_waves(self):
        test_data = \
            """
		    FirmID                                         VendorName  ACSI_14  ACSI_15  ACSI_16  ACSI_14_BENCH  ACSI_15_BENCH  ACSI_16_BENCH
		0       1                        Apple [Cellular Telephones]       79     80.0     81.0             78           78.0           79.0
		1       1                         Apple [Personal Computers]       84     84.0      NaN             78           77.0            NaN
		2       2   Google [Internet Search Engines and Information]       83     78.0      NaN             80           76.0            NaN
		3       2                    Google+ [Internet Social Media]       84      NaN      NaN             71            NaN            NaN
		4       2           YouTube (Google) [Internet Social Media]       78     76.0     77.0             71           74.0           73.0
		5       3  Bing (Microsoft) [Internet Search Engines and ...       73     72.0     75.0             80           76.0           77.0
		6       3                      Microsoft [Computer Software]       75     75.0     80.0             76           74.0           81.0
		t
		   FirmID                                         VendorName  ACSI_14  ACSI_15  ACSI_16  ACSI_16_PADDED  ACSI_14_BENCH  ACSI_15_BENCH  ACSI_16_BENCH  ACSI_16_BENCH_PADDED
		0       1                        Apple [Cellular Telephones]       79     80.0     81.0              81             78           78.0           79.0                    79
		1       1                         Apple [Personal Computers]       84     84.0      NaN              84             78           77.0            NaN                    77
		2       2   Google [Internet Search Engines and Information]       83     78.0      NaN              78             80           76.0            NaN                    76
		3       2                    Google+ [Internet Social Media]       84      NaN      NaN              84             71            NaN            NaN                    71
		4       2           YouTube (Google) [Internet Social Media]       78     76.0     77.0              77             71           74.0           73.0                    73
		5       3  Bing (Microsoft) [Internet Search Engines and ...       73     72.0     75.0              75             80           76.0           77.0                    77
		6       3                      Microsoft [Computer Software]       75     75.0     80.0              80             76           74.0           81.0                    81
		"""		
        ofn = 'test_pad_original_from_ACSI_clean_and_augmented.csv' 
        vendor = 'test_vendor'
        self.orig = pd.read_csv(os.path.join(os.path.abspath(path), vendor, ofn))			

        tfn = 'test_pad_target_ACSI_clean_and_augmented_2_waves.csv'
        self.target = pd.read_csv(os.path.join(os.path.abspath(path), vendor, tfn))

        t = self.target
        o = self.orig		

        result = pad(self.orig, waves_to_update = 2, vendor = 'ACSI', wave = 2016)

        for col, dtype in zip(result.columns, result.dtypes):			
            if not dtype == object:
                diff = result[col].round(5) - self.target[col].round(5)
                self.assertFalse(diff.any())		


##Need Larry's z-scores; need ratios; need merge test	

class TestAggregate(unittest.TestCase):
    """Test Panda's aggregation against SPSS's from 2016 data"""
    def setUp(self):
        """grab input and target docs"""
        verbose = False
        ofn = 'test_atomic_16.csv'
        vendor = 'test_vendor'
        self.target = pd.read_csv(os.path.join(os.path.abspath(path), vendor, ofn))

        tfn = 'test_data_acsi_augmented_16.csv'
        self.orig = pd.read_csv(os.path.join(os.path.abspath(path), vendor, tfn))

        t = self.target
        o = self.orig

    def test_aggregate(self):
        "Ensure aggregate() works as expected"
        test_data = \
            """
		t (from 'Atomic Clean Jan 9 2017.csv')
		   FirmID            CompanyName  ACSI_15_BENCH  ACSI_16_BENCH
		0       1             Apple Inc.      77.500000      78.500000
		1       2           Google, Inc.      74.666667      74.333333
		2       3  Microsoft Corporation      76.000000      78.500000

		o (from 'ACSI Clean and Augmented)
		   FirmID            CompanyName  ACSI_ID                                  Dec 2016 ASCIName  ACSI_15_BENCH  ACSI_16_BENCH  ACSI_16_BENCH_PADDED
		0       1             Apple Inc.       37                        Apple [Cellular Telephones]             78             79                    79
		1       1             Apple Inc.       38                         Apple [Personal Computers]             77             78                    78
		2       2         Alphabet, Inc.      201   Google [Internet Search Engines and Information]             76             77                    77
		3       2         Alphabet, Inc.      202                    Google+ [Internet Social Media]             74             73                    73
		4       2         Alphabet, Inc.      485           YouTube (Google) [Internet Social Media]             74             73                    73
		5       3  Microsoft Corporation       68  Bing (Microsoft) [Internet Search Engines and ...             76             77                    77
		6       3  Microsoft Corporation      284                      Microsoft [Computer Software]             74             81                    81
		7       3  Microsoft Corporation      285     Microsoft Mobile (Nokia) [Cellular Telephones]             78             79                    79
		8       3  Microsoft Corporation      294  MSN (Microsoft) [Internet Search Engines and I...             76             77                    77

		"""
        result = aggregate(self.orig)
        #make sure all the non-text columns are identical
        for col, dtype in zip(result.columns, result.dtypes):			
            if not dtype == object:
                diff = result[col].round(5) - self.target[col].round(5)
                self.assertFalse(diff.any())




    def test_ratios(self):
        pass

    def test_zscores(self):
        pass

    def test_means(self):
        pass



class TestQAFunctions(unittest.TestCase):
    
    def test_check_for_new_company(self):
        
        id_parent = {
            1 : {'company': u'Apple Inc.', 'ticker': 'AAPL'}, 
            2 : {'company': u'Alphabet Inc.', 'ticker': 'GOOGL'}, 
            3 : {'company': u'Microsoft Corp.', 'ticker': 'MSFT'},
            4 : {'company': u'Exxon Mobil Corp.', 'ticker': 'XOM'}, 
            }
        
        #Source (vendor) data
        data = [
               [1, 'Apple Inc.', 10, 20, 30],
               [2, 'Google, Inc.', 20, 30, 40],
               [3, 'Microsoft Corporation', 20, nan, 40] ,  #new and in popfr
               [4, 'Exxon Mobil Corporation', 20, nan, 40], #new and in popfr
               [9999, 'Test', 9, 9, 9]                      #new and not in popfr
               ]
        
        columns = ['FirmID', 'CompanyName',  'c1', 'c2', 'c3']
        index = 'FirmID'
        sdf = pd.DataFrame(data, columns = columns)        
        sdf.set_index(index, inplace = True)
        
        #Destination (roll-up table, maybe)
        data = [
               [1, 'Apple Inc.', 10, 20, 30],
               [2, 'Google, Inc.', 20, 30, 40],
               #[3, 'Microsoft Corporation', 20, nan, 40] ,  #new and in popfr
               #[4, 'Exxon Mobil Corporation', 20, nan, 40], #new and in popfr
               #[9999, 'Test', 9, 9, 9]                      #new and not in popfr
               ]
        ddf = pd.DataFrame(data, columns = columns)        
        ddf.set_index(index, inplace = True)
        
        new_not_in_pop_frame = check_for_new_company(sdf = sdf, ddf = ddf, verbose = False)
        
        #if this works we should have only the rogue 9999 FirmID reported
        self.assertEqual(list(new_not_in_pop_frame.index), [9999])

    
    def test_check_company_firmid(self):
        """Checks whether the company / firmid pairings in vendor data are still good.
             Potential issues:
                - a buyout has reassigned a company to a different parent;
                - the company name is mismatched to the Drucker version"""
        
        id_parent = {
            1 : {'company': u'Apple Inc.', 'ticker': 'AAPL'}, 
            2 : {'company': u'Alphabet Inc.', 'ticker': 'GOOGL'}, 
            3 : {'company': u'Microsoft Corp.', 'ticker': 'MSFT'},
            4 : {'company': u'Exxon Mobil Corp.', 'ticker': 'XOM'}, 
            }
        
        data = [
               [1, 'Apple Inc.', 10, 20, 30],
               [2, 'Google, Inc.', 20, 30, 40],
               [3, 'Microsoft Corporation', 20, nan, 40] ,  #new and in popfr
               [4, 'Exxon Mobil Corporation', 20, nan, 40], #new and in popfr
               [9999, 'Test', 9, 9, 9]                      #new and not in popfr
               ]
        
        columns = ['FirmID', 'CompanyName',  'c1', 'c2', 'c3']
        index = 'FirmID'
        sdf = pd.DataFrame(data, columns = columns)        
        sdf.set_index(index, inplace = True)
        
        mismatched_names = check_company_firmid(sdf)


            
        pass

    def test_test_outliers(self):
        """Ensure check_distributional_outliers() working as expected."""
        
        data = [
               ['firm1', 'text', 30, 20, .1],   #contains outlier
               ['firm2', 'text', 20, 30, 40],
               ['firm3', 'text', 20, nan, 40],
               ['firm4', 'text', 20, 25, 40],
               ['firm5', 'text', 25, 49, 45],
               ['firm6', 'text', 25, 34, 45],
               ['firm7', 'text', 25, 50, 45],
               ['firm8', 'text', 40, nan, 40],
               ['firm9', 'text', 50, nan, 40],
               ['firm10','text', 60, nan, 40],
               ['firm11', 'text',60, nan, 40],
               ['firm12', 'text',60, nan, 40],
               ['firm13', 'text',60, nan, 40],
               ['firm14', 'text',60, nan, 40], 
               ]
        
        """
                     Zc1       Zc2       Zc3
        firm                                
        firm1  -0.544764 -1.180852 -3.412075
        firm2  -1.109705 -0.375726  0.159192
        firm3  -1.109705       NaN  0.159192
        firm4  -1.109705 -0.778289  0.159192
        firm5  -0.827235  1.154014  0.606719
        firm6  -0.827235 -0.053675  0.606719
        firm7  -0.827235  1.234527  0.606719
        firm8   0.020176       NaN  0.159192
        firm9   0.585117       NaN  0.159192
        firm10  1.150058       NaN  0.159192
        firm11  1.150058       NaN  0.159192
        firm12  1.150058       NaN  0.159192
        firm13  1.150058       NaN  0.159192
        firm14  1.150058       NaN  0.159192
        """        
        
        df = pd.DataFrame(data, columns = ['firm', 'c0', 'c1', 'c2', 'c3'])
        index = 'firm'
        df.set_index(index, inplace = True)    

        data = [
               ['firm1', 'text', nan, nan, .1],
               ]
        
        expected = pd.DataFrame([['firm1', nan, nan, .1]], columns = ['firm', 'c1', 'c2', 'c3'])
        index = 'firm'
        expected.set_index(index, inplace = True)        
        
        bad = test_outliers(df, threshhold = 3, verbose = False, first_data_col= 1)

        self.assertEqual(str(bad), str(expected))
        a = 1
        
    
    def test_get_firmid_from_vendor_name(self):
        """Given a Series of vendor parent names, use the vendor dict to look up the Drucker parent name,
          then use the parent_id dict from py_dictionaries.py to return the FirmID.
          """
        
        parent_id = {
            "Praxair, Inc." : {'ticker': 'ISRG', 'firmID': 153}, 
             "Visa Inc." : {'ticker': 'MGA', 'firmID': 18},
            }
        
        vend_dict =  {	"Praxair, Inc." : {'Industry': 'Materials', 'ParentCompanyName': u'Praxair, Inc.'}, 
                        "Visa International" : {'Industry': 'Software & Services', 'ParentCompanyName': u'Visa Inc.'}, 
                      }
        ser = pd.Series(data = ["Praxair, Inc.",  "Visa International"])
        ids = get_firmid_from_vendor_name(ser, vend_dict, parent_id_dict = parent_id)
        self.assertEqual(list(ids), [153, 18])
        a = 1
        
    def test_fill_in_firmid_gaps(self):
        """Scenario:  we have some FirmIDs associated with vendor companies that we've gotten
            by mapping these to those in the official population frame.  But the vendor
            has supplied a superset of these, along with an already-identified set of
            FirmIDs.   We'll fill in any gaps left by the original mapping (partial_ser)
            with vendor-supplied FirmIDs acquired as part of creating their vend_dict
            (provided in dictionary.xlsx)."""
        
        ser = pd.Series(['acorp', 'bcorp', 'ccorp', 'dcorp'])
        partial_ser = pd.Series([666, nan, nan, 999])
        vend_dict = {"acorp" : {'Industry': 'ind', 'ParentCompanyName': u'a', firmid_col: 1},
                     "bcorp" : {'Industry': 'ind', 'ParentCompanyName': u'b', firmid_col: 2},
                     "ccorp" : {'Industry': 'ind', 'ParentCompanyName': u'c', firmid_col: 3},
                     "dcorp" : {'Industry': 'ind', 'ParentCompanyName': u'd', firmid_col: 4},
                     }
        
        fixed = fill_in_firmid_gaps(ser, vend_dict, partial_ser, verbose = False)
        
        self.assertEqual(list(fixed), [666, 2, 3, 999])
        

        
            
            
        
class TestPandasFunctions(unittest.TestCase):
    def test_aggregate_append(self):
        """Input dataframe and a break column.  Returns a dataframe w/new column, containing
            aggregated averages.   For instance, if we needed to roll industry averages and
            assign the average as a new column, this is the go-to routine."""
        data = [
               [1, 'firm1', 'ind1', 10, 20, 30],
               [2, 'firm2', 'ind1', 20, 30, 40],
               [3, 'firm3', 'ind2', 20, nan, 40],
               [4, 'firm3', 'ind2', 40, nan, 20],
               
            ]
        
        df = pd.DataFrame(data, columns = ['FirmID', 'company', 'ind', 'c1', 'c2', 'c3'])
        index = 'FirmID'
        df.set_index(index, inplace = True)
        
        expected_data = \
            [['firm1', 10, 20.0, 30 , 15 , 25.0, 35],   #last 3 colsreflect ind means
             ['firm2', 20, 30.0, 40, 15, 25.0, 35], 
             ['firm3', 20, NaN, 40, 30, NaN, 30], 
             ['firm3', 40, NaN, 20, 30, NaN, 30 ] ] 
            
        expected_cols =  ['company',  'c1' ,  'c2', 'c3' , 'c1_ind_mean',  'c2_ind_mean' , 'c3_ind_mean' ]
        
        """\
               company  c1    c2  c3  c1_ind_mean  c2_ind_mean  c3_ind_mean
        FirmID                                                             
        1        firm1  10  20.0  30           15         25.0           35
        2        firm2  20  30.0  40           15         25.0           35
        3        firm3  20   NaN  40           30          NaN           30
        4        firm3  40   NaN  20           30          NaN           30
        """
        
        index_col = 'ind'
        rollup_suffix = '_ind_mean'
        first_data_col = 2        
        
        expected = pd.DataFrame(data = expected_data, columns= expected_cols)
        
        result =  aggregate_append(df = df, index_col = index_col,
                                   first_data_col = first_data_col, rollup_suffix = '_mean')
        
        self.assertAlmostEqual(result.iloc[0, 4], 15)
        self.assertAlmostEqual(result.iloc[0, 5], 25)
        self.assertAlmostEqual(result.iloc[0, 6], 35)
        self.assertAlmostEqual(result.iloc[2, 4], 30)
        #self.assertTrue(result.iloc[2, 5] is nan)
        self.assertAlmostEqual(result.iloc[2, 6], 30)
        
        a = 1
        


        
        
        
    
    def test_average_columns(self):
        """Ensure pivot_table() working as expected."""
        
        data = [
               ['firm1', 10, 20, 30],
               ['firm2', 20, 30, 40],
               ['firm3', 20, nan, 40]   
               ]
        
        df = pd.DataFrame(data, columns = ['firm', 'c1', 'c2', 'c3'])
        index = 'firm'
        df.set_index(index, inplace = True)
        
        cols = ['c1', 'c2', 'c3']
        
        new_col_name = 'my_mean'
        avg = average_cols(df = df, cols = cols)
        self.assertEqual(list(avg.values), [20, 30, 30])
        
        
if __name__ == '__main__':
    unittest.main()
    #write_scratch_qa_files_for_augmentation()