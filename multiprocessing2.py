import os
import multiprocessing
import time

def cube(i):
    print("calculating cube.....")
    for n in i:
        result = n * n * n
        time.sleep(5)

        print(f"the cube of {n} is {result}")


def square(i):
    print("calculating square.....")
    for n in i:
        result = n * n
        time.sleep(5)
        print(f"the square of {n} is {result}")



if __name__ == '__main__':
    a = range(10)
    t = time.time()
    p1 = multiprocessing.Process(target=cube, args=(a,))
    p2 = multiprocessing.Process(target=square, args=(a,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    print("completed at :", time.time() - t)
