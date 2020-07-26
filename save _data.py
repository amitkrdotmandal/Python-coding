import pandas as pd
from numpy import *

"""
p1 = ([1, 5, 6, 4, 5, 2, 7, 8, 9, 6, 8, 90, 34])
data=pd.read_csv('C:\\Users\\amitk\\OneDrive\\Desktop\\asdf.csv')
f = pd.DataFrame({'name': p1})
f.to_csv('C:\\Users\\amitk\\OneDrive\\Desktop\\asd1.csv')
"""
import xlsxwriter

p1 = matrix([1, 5, 6, 4, 5, 2, 7, 8, 9, 6, 8, 9, 34,23])
p2 = matrix([1, 5, 6, 4, 5, 2, 7, 8, 9, 6, 8, 90, 34, 67])
outworkbook = xlsxwriter.Workbook("C:\\Users\\amitk\\OneDrive\\Desktop\\amit.xlsx")
outsheet=outworkbook.add_worksheet()
outsheet.write("A1", "p1")
outsheet.write("B1", "p2")

outworkbook = xlsxwriter.Workbook("C:\\Users\\amitk\\OneDrive\\Desktop\\amit.xlsx")
outsheet=outworkbook.add_worksheet()
outsheet.write("A1", "p1")
outsheet.write("B1", "p2")

a, b=p1.shape
for i in range(0, b):
    outsheet.write(i+1, 0, p1[0,i])
    outsheet.write(i+1, 1, p2[0,i])
outworkbook.close()




