from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
ax=plt.axes(projection="3d")
p = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(p)


p1 = [1, 5, 6, 4, 5, 2, 7, 8, 9]
print(p1)
p2 = [1, 5, 6, 4, 5, 2, 7, 8, 9]
ax.scatter(p, p1, p2)

plt.show()

