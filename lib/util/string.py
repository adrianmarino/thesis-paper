import re

def between(value, start, end):
    return value[value.find(start)+len(start)+1:value.find(end)]


def str_join(items, separator=', ', last_separator=' and ', last_word='.'):
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    result = separator.join(str(item).strip() for item in items[:-1])
    result += last_separator + str(items[-1]).strip()
    return result + last_word