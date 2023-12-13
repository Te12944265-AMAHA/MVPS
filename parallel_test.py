import math
import numpy as np
from timebudget import timebudget
from multiprocessing import Pool
import os
from tqdm import tqdm
#import ray

#ray.init()

def init_worker(data):
    # declare scope of a new global variable
    global shared_data
    # store argument in the global variable for this process
    shared_data = data

#@ray.remote
def complex_operation(args):
    input_index, _ = args
    global shared_data
    iterations_count = round(1e7)
    #print("Complex operation. Input index: {:2d}\n".format(input_index))
    data = np.ones(iterations_count)*input_index
    res = np.exp(data) * np.sinh(data)
    ret = np.zeros(300)
    ret[input_index] = res[input_index]
    return input_index, ret

@timebudget
def run_complex_operations(operation, input, pool):
    #ray.get([operation.remote(i) for i in input])
    # dispatch parallel tasks. each task takes in shared global vars and some
    # patch-dependent input arguments, returns an output. 
    # then, assemble the output
    data = list(tqdm(pool.imap_unordered(operation, input), total=len(input)))
    #data = pool.map(operation, input)
    assembled = np.zeros((100, 300))
    for ret in data:
        assembled[ret[0], :] = ret[1]
    print(assembled[:10,:10])

if __name__ == "__main__":

    processes_count = 10
    arr = np.ones((10, round(1e7)))
    processes_pool = Pool(processes_count, initializer=init_worker, initargs=(arr,))
    input = [(i, i*3) for i in range(100)]
    #run_complex_operations(complex_operation, input)
    run_complex_operations(complex_operation, input, processes_pool) 
    #print(arr[:,0])