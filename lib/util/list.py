import numpy as np
from itertools import compress, product

def combinations(items):
    combinations_list = ( set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )
    return set(list([frozenset(it) for it in combinations_list if len(it) == 2]))


def subtract(a, b):
    return list(set(a) - set(b))



def empty(list):
    return list is None or len(list) == 0


def nanmean(arr, axis=0):
    arr = arr.T if axis==0 else arr
    
    result = []    
    for axi_values in arr:
        axi_values = axi_values[axi_values != 0]
        if len(axi_values) > 0:
            result.append(np.mean(axi_values))
    return np.array(result)


def nanmedian(arr, axis=0):
    arr = arr.T if axis==0 else arr
    
    result = []    
    for axi_values in arr:
        axi_values = axi_values[axi_values != 0]
        if len(axi_values) > 0:
            result.append(np.median(axi_values))
    return np.array(result)