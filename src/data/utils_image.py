import base64
import numpy as np
import cv2
from PIL import Image

#####################################################################################
### Functions to transform image data types
#####################################################################################

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