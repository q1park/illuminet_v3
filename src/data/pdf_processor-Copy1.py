import os
import re
import glob

import numpy as np
import pandas as pd

from src.data.structures import Processor, LineData
from src.data.utils import group_df_dict
from src.data.pdf_reader import pdf_to_spans, spans_to_lines, shift_boxes

class PDFProcessor(Processor):
    def __init__(self, data_dir, **kwargs):
        super(PDFProcessor, self).__init__(**kwargs)
        self.dir = os.path.join(*re.split(r'[\/\\]', data_dir))
        self.pdf_path = glob.glob(os.path.join(self.dir,'*.pdf'))[0]
        
        self.df = None
        self.images = None
        
    def load_pdf(self, fix_boxes=False, verbose=False):
        span_df, lines_boxes, self.images =  pdf_to_spans(self.pdf_path)
        self.df = spans_to_lines(span_df, lines_boxes)
                
        if fix_boxes:
            self.fix_boxes()
            
    def fix_boxes(self):
        lines_boxes = shift_boxes(group_df_dict(self.df, 'page', *['x0', 'y0', 'x1', 'y1']))
        self.df.update(pd.DataFrame([list(x) for x in lines_boxes.values()], columns = ['x0', 'y0', 'x1', 'y1']))
            
    def add_feature(self, func_df, *names):
        if len(names)==1:
            lines = func_df(self.df)
            self.df[names[0]] = lines
        else:
            for name, lines in zip(names, func_df(self.df)):
                self.df[name] = lines