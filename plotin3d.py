""""import pandas as pd
data1 = pd.read_excel("C:\\Users\\amitk\\OneDrive\\Desktop\\amit.xlsx",sheet_name='Sheet2')
data2=data1(1,2)
print(data2)"""
import xlrd
from numpy import *
import matplotlib.pyplot as plt
import time
path ="C:\\Users\\amitk\\OneDrive\\Desktop\\amit.xlsx"
workbook=xlrd.open_workbook(path)
worksheet=workbook.sheet_by_index(1)
rows=worksheet.nrows
columns=int((worksheet.ncols)/3)
print(columns)
XYZ = matrix([[0, 0.0, 0],
            [1, 0, 0],
            [1, 1.0, 0],
            [0, 1.0, 0]])
noNode, DFPN = XYZ.shape


X=[]
Y=[]
Z=[]
for i in range(0, noNode):
    X.append(XYZ[i,0])
    Y.append(XYZ[i, 1])
    Z.append(XYZ[i, 2])

print(X)
print(Y)
print(Z)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i in range(0, columns):
    Ux=[]
    Uy=[]
    Uz=[]
    for j in range(0, rows):
        ux=worksheet.cell_value(j, 3*i)
        uy = worksheet.cell_value(j, 3 * i+1)
        uz = worksheet.cell_value(j, 3 * i + 2)
        Ux.append(ux)
        Uy.append(uy)
        Uz.append(uz)
        #print(ux)
        #print(uy)
       # print(uz)
        #print('yyy')
    x =Ux
    y = Uy
    z = Uz
    for j in range(0, rows):
        x[j]=X[j]+Ux[j]
        y[j] = Y[j] + Uy[j]
        z[j] = Z[j] + Uz[j]
    print(x)
    print(y)
    print(z)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")

    #ax = fig.axes(projection="3d")


    ax.plot(x, y, z,'ro')
    plt.show(block=False)

    #plt.draw()
    plt.pause(.1)
    plt.cla()


    print('xxx')

#ax.scatter(x, y, z)


