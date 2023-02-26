import math


def round_all(values, decimals=None):
    return [round_(v, decimals)for v in values]


def round_(value, decimals=None):
    return round(value, decimals) if decimals is not None and decimals >= 0 else value


def clamp(value, min_val=0, max_val=math.inf):
    if value <= min_val:
        return min_val
    elif value >= max_val:
        return max_val
    else:
        return value