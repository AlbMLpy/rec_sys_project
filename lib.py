from itertools import takewhile
import numpy as np
from numba import njit

@njit
def index(array, item):
    '''.Checked.'''
    for idx, val in enumerate(array):
        if val == item:
            return idx

def searchranked(array, target_pos, topk):
    '''
    Finds a rank of target (located at `target_pos`) in `array`
    Returns rank value if target's among topk elements, otherwise returns 0.
    Pessimizes rank if there're other elements in `array` equal to target.
    Note: rank values start from 1, meaning "the first among topk elements".
    '''
    target = array[target_pos]
    vals = takewhile(# select no more than topk elemenets with values >= target value
        lambda x: x[0]<=topk, # for identical scores, `<=` pushes target out of top-k list
        enumerate(a for a in array if a >= target) # allows identical scores to push target out
    )
    pos = len(list(vals)) # always >= 1, because `target` itself is in `array`
    return pos if pos <= topk else 0


def topsort(a, topk):
    """
        To fill in!
    """
    parted = np.argpartition(a, -topk)[-topk:]
    return parted[np.argsort(-a[parted])]
