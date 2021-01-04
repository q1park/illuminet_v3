from collections import Counter
from src.data.structures import LineData


#####################################################################################
### Functions to make line spacing feature
#####################################################################################

def make_doc_mask(image, structure):
    common_images = [
        img for img, count in Counter(
            [x for x in image.lines if x is not None]
        ).most_common()[:20]
        if count/len(structure.data)>0.2
    ]
    mask_lines = [0 if x not in common_images else 1 for x in image.lines]
    return LineData(name='mask', lines=mask_lines)

#####################################################################################
### 
#####################################################################################