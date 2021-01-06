import sys
import re
import numpy as np
from collections import Counter

def compute_entropy(text):
    word_list = [x for x in text.split() if len(x)>0]
    probas = {k:v/len(word_list) for k,v in Counter(word_list).items()}
    return sum(v*np.log2(1/v) for v in probas.values())

def group_df_dict(df, by, *args):
    if len(args)==0:
        args = tuple(df.columns)
    return {k:list(v[list(args)].to_records(index=False)) for k,v in df[[by]+list(args)].groupby(by)}

def reget(pattern, string):
    regex = re.search(pattern, string)
    return regex.group() if regex is not None else None

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

## recursively compute size of object

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Mark as seen *before* entering recursion to  handle self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def jaccard_list(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def jaccard_matrix(dict_of_lists, rnd=None):
    jaccard = np.zeros((len(dict_of_lists),)*2)
    rows = np.array(list(dict_of_lists.keys()))
    
    for i in range(len(dict_of_lists)):
        for j in range(len(dict_of_lists)):
            jaccard[i,j] = jaccard_list(dict_of_lists[rows[i]], dict_of_lists[rows[j]])
    if rnd is not None:
        jaccard = np.around(jaccard, rnd)
    return jaccard, rows

#####################################################################################
### Functions to select bounding-box edges
#####################################################################################

def box_edges(boxes, *args, rnd=None):
    if len(args)==0:
        if rnd:
            return [np.around(tuple(b), rnd) for b in boxes]
        else:
            return [tuple(b) for b in boxes]
    elif len(args)==1:
        if rnd:
            return [np.around(b[args[0]], rnd) for b in boxes]
        else:
            return [b[args[0]] for b in boxes]
    else:
        if rnd:
            return [tuple(np.around(b[idx], rnd) for idx in args) for b in boxes]
        else:
            return [tuple(b[idx] for idx in args) for b in boxes]
    
def box_counter_page(box, *args, rnd=None):
    return {n:Counter(box_edges(b, *args, rnd=rnd)) for n,b in box.items()}

def box_counter(box, *args, rnd=None):
    return Counter([x for n in box for x in box_edges(box[n], *args, rnd=rnd)])