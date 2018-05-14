import unittest
import pandas as pd

def x():
	pass

def add_dfs(df1 = None, df2 = None):
	to_return = df1 + df2
	return to_return

def merge_dfs_flex ( *dfs):
	pass

class TestMe(unittest.TestCase):
	def xtest_add(self):
		self.assertEqual(merge_dfs()[0], merge_dfs()[1])
	def test_merge_dfs(self):
		df1 = pd.Series([1, 2, 3])  #column object (a stack = pd.DataFrame)
		df2 = pd.Series([10, 10, 10])
		#self.assertEqual(add_dfs(df1, df2), pd.Series(11, 12, 14))
		self.assertEqual(list(add_dfs(df1, df2).values),
		                 list(pd.Series([11, 12, 13]).values)
		                 )
		
		
	
if __name__ == "__main__":
	unittest.main()
