
import xlrd
from numpy import *
import matplotlib.pyplot as plt
import time
path ="C:\\Users\\amitk\\OneDrive\\Desktop\\amit1.xlsx"     ##################################################################3
workbook=xlrd.open_workbook(path)
worksheet=workbook.sheet_by_index(1)
rows=worksheet.nrows
columns=int((worksheet.ncols)/3)
print(columns)


pth ="C:\\Users\\amitk\\OneDrive\\Desktop\\F1.xlsx"         ##################################################################
workbookk1=xlrd.open_workbook(pth)


worksheet_element=workbookk1.sheet_by_index(0)
rows_ele=worksheet_element.nrows
columns_ele=worksheet_element.ncols
element = ones((rows_ele, columns_ele), dtype = int)
for i in range(0, rows_ele):
    for j in range(0, columns_ele):
        element[i, j] = int(worksheet_element.cell_value(i, j))
#print(element)
noEle, NPE = element.shape

worksheet_node=workbookk1.sheet_by_index(1)
rows_node=worksheet_node.nrows
columns_node=worksheet_node.ncols
XYZ= ones((rows_node, columns_node))
for i in range(0, rows_node):
    for j in range(0, columns_node):
        XYZ[i, j] = worksheet_node.cell_value(i, j)
#print(nodeCordinate)



"""
XYZ = matrix([[0, 0.0, 0],
                        [1, 0, 0],
                        [1.3, 1.2, 0],
                        [0.2, 1.4, 0],
                        [1, 2, 0],
                        [0, 2.0, 0]])
"""
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
for i in range(0, columns, 1):
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
    for ele in range(0, noEle):
        eleNode1 = element[ele, 0]
        eleNode2 = element[ele, 1]
        eleNode3 = element[ele, 2]
        eleNode4 = element[ele, 3]

        Xo1 = x[eleNode1 - 1]
        Xo2 = x[eleNode2 - 1]
        Xo3 = x[eleNode3 - 1]
        Xo4 = x[eleNode4 - 1]
        Yo1 = y[eleNode1 - 1]
        Yo2 = y[eleNode2 - 1]
        Yo3 = y[eleNode3 - 1]
        Yo4 = y[eleNode4 - 1]
        Zo1 = z[eleNode1 - 1]
        Zo2 = z[eleNode2 - 1]
        Zo3 = z[eleNode3 - 1]
        Zo4 = z[eleNode4 - 1]
        XXo = [Xo1, Xo2, Xo3, Xo4, Xo1]
        YYo = [Yo1, Yo2, Yo3, Yo4, Yo1]
        ZZo = [Zo1, Zo2, Zo3, Zo4, Zo1]
        ax.plot(XXo, YYo, ZZo)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show(block=False)
    plt.pause(2)
    plt.cla()



from mpl_toolkits.mplot3d import Axes3D
# Make data.
