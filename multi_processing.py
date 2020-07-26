import os
from multiprocessing import Process, current_process
def swer(n):
    result=n*n*n
    #process_id=os.getpid()
    #print(f"process id is {process_id}")
    process_name=current_process().name
    print(f"process name: {process_name}")
    print(f"the cube of {n} is {result}")

if __name__ == '__main__':
    a=range(200)
    processes=[]
    for ai in a:

        process=Process(target=swer, args=(ai,))
        processes.append(process)
        process.start()