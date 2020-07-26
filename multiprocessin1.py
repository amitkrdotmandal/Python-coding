import os
from multiprocessing import Process, current_process
import time
import threading
def cube(i):
    print("calculating cube.....")
    for n in i:
        result = n * n * n
        time.sleep(.2)

        print(f"the cube of {n} is {result}")


def square(i):
    print("calculating square.....")
    for n in i:
        result = n * n
        time.sleep(.2)
        print(f"the square of {n} is {result}")




a=range(10)
t=time.time()
t1=threading.Thread(target=cube, args=(a,))
t2=threading.Thread(target=square, args=(a,))

t1.start()
t2.start()

t1.join()
t2.join()
print("completed at :", time.time()-t)