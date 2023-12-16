import re

def between(value, start, end):
    idx1 = value.index(start)
    idx2 = value.index(end)

    return ''.join([value[idx] for idx in range(idx1 + len(start) + 1, idx2)])
