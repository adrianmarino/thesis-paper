import re

def between(value, start, end):
    return value[value.find(start)+len(start)+1:value.find(end)]