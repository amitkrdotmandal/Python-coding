
from multiprocessing import Pool
from numpy import *

import time
a = matrix([-1 / sqrt(3), 1 / sqrt(3)])
b = matrix([-1 / sqrt(4), 1 / sqrt(4)])
c = matrix([-1 / sqrt(5), 1 / sqrt(5)])
d = matrix([-1 / sqrt(6), 1 / sqrt(6)])

work1 = ([["A", 5], ["as", 1, 8], [a]], [["B", 2], ["bs", 5, 6], [b]], [["C", 1], ["cs", 4, 2], [c]], [["D", 3], ["ds", 7, 1], [d]])


def work_log(work_data):
    print(" Process %s waiting %s seconds" % (work_data[0][0], work_data[0][1]))
    #print(work_data[1][0])
    #time.sleep(int(work_data[0][1]))
    #print(" Process %s Finished." % work_data[0][1])
    UtVtWttranspose = work_data[2][0].transpose()
    srMatrixUtVtWttranspose = UtVtWttranspose.dot(work_data[2][0])
    return srMatrixUtVtWttranspose


def pool_handler():
    p = Pool()
    result = p.map(work_log, work1)
    print(result[1][0, 1])


if __name__ == '__main__':
    pool_handler()
    #a=work1[3]
    #print(a[2][0][0,0])

