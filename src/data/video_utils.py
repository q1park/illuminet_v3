import numpy as np

def rect_to_bb(rect):
    # convert dlib bounding box to (x, y, w, h) as in opencv
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    
    for i in range(0, 68): # convert 68 landmarks to (x, y)-coordinates
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])