from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#array
"""
arr=array([2.01,1])
arr1=linspace(2.01,1,10)
arr3=linspace(0,0,10)
arr=array([2.01,1])
ar1=array([2.01,5])

arr[1]=11
ar2=arr+ar1

ar5=array([ ])

ar6=concatenate([ar1,ar2])
arCopy=ar6.copy()
arCopy[2]=144
print(arr[1])
print(arr3)
print(sqrt(ar6))
print(sqrt(arCopy))
"""

#m1=matrix('1,34,5;5,6,32;5,61,32')
#m2=matrix('1,34,11;5,6,304;12,6,32')
#m3=m1*m2
#m3[0,0]=121
#m4=linalg.inv(m3)  #to perform inverse  of matrix
#t,p=m4.shape       #shape of matrix
#print(m3[0,0])
#print(m3)
#print(m4)
#print(t)


p = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(p)

p1 = [1, 5, 6, 4, 5, 2, 7, 8, 9]
print(p1)
#p2 = p*3
plt.plot(p, p1)
#ax = plt.axes(projection="3d")
plt.show()
print(p)


