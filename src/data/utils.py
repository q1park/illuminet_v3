import sys
import base64
import re
import json
import numpy as np
from collections import Counter

import cv2
from PIL import Image

## load and save json

def reget(pattern, string):
    regex = re.search(pattern, string)
    return regex.group() if regex is not None else None

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

## load and save json

def int_keys(ordered_pairs):
    result = {}
    for key, value in ordered_pairs:
        try:
            key = int(key)
        except ValueError:
            pass
        result[key] = value
    return result

def json_load(path):
    with open(path, 'r') as f: 
        pydict = json.loads(f.read(), object_pairs_hook=int_keys)
    return pydict

def json_save(pydict, path):
    with open(path, 'w') as f: 
        f.write(json.dumps(pydict))
        
## image conversions

def imgpath2np(imgpath):
    return np.array(Image.open(imgpath).convert('RGB'))

def bimg2utf(bimg):
    return base64.b64encode(bimg).decode('utf8')

def utf2bimg(uimg):
    return base64.b64decode(uimg.encode('utf8'))

def bimg2np(bimg):
    return cv2.imdecode(np.frombuffer(bimg, np.uint8), -1)

def np2bimg(npimg, encoding=None):
    if encoding is None:
        encoding = '.png' if npimg.shape[-1]==4 else '.jpg'
    return cv2.imencode(encoding, npimg)[1].tobytes()

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