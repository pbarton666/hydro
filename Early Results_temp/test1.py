import pandas as pd
import numpy as np
filename = open('CouleeflowT.csv','r')
df = pd.read_csv(filename)
filename2 = open('Flow Revised.xlsx','r')
df2 = pd.read_excel(filename2)
