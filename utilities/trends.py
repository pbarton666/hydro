#trends.py

import unittest

def calc_trend(values = None):
	""" Calculate trend as Pearson product-moment correlation coefficient
	    (analog to Excel CORREL function cf.
		http://www.excelfunctions.net/Excel-Correl-Function.html).

		One wrinkle:  when the final two years' data are identical, assume that
		the last data point is a placeholder - the underlying data have not been
		updated for the latest wave.   In this case, we'll calculate the trend on
		(n-1) data points, dropping the last value.  This has the effect of
		extrapolating the last 'good' trend to the current analysis."""

	#last 2 values the same?
	if values[-1] == values[-2]:
		#... if so calculate trend based on only 'good' values
		index = list(range(1, len(values)))      #-->list [ 1.. length of values -1]
		values = values[:-1]		
	else:
		index = list(range(1, len(values) + 1))  #-->list [ 1.. length of values]

	#calcualte the correlation matrix
	cor_matrix = numpy.corrcoef(values, index)

	#return one of the off-diagonal values (the 'trend')	
	return cor_matrix[0, 1]


class TestTrend(unittest.TestCase):
	def test_trend(self):
		"""test Excel's CORREL function analog using 'Profiling Template Jan. 25 2018.xlsx'
		   data from OVERALL INDEX for Drucker (Row 2), OVERALL INDEX for Customer (Row 3), and
		   Economic Spread (Row 44).
		   """
		test = [[ (67.5693,71.7342,70.9711,70.4216,69.4494,69.4494), 0.240755],    #consumer overall
		        [ (56.7761,60.0372,57.9643,56.6113,54.0998,54.4190), -0.7443612],  #econ spread
		        [ (82.9365,84.4830,81.3434,83.3056,81.4903,80.7020), -0.678335],   #drucker overall

		        ]      
		for values, expected in test:
			trend =  calc_trend(values)
			self.assertAlmostEqual(calc_trend(values), expected, places = 5)



if __name__ == '__main__':

	unittest.main()