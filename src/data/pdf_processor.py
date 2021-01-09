import os
import re
import glob

import numpy as np
import pandas as pd

from src.data.utils_io import save_pickle, load_pickle
from src.data.pdf_featurizer import (
    make_caption, make_mask, make_spacing_back, make_chunks, make_sections, make_image_urls
)
from src.data.pdf_reader import pdf_to_spans
#from src.data.tricks import make_chunk_summaries, make_group_summaries, make_section_summaries
from src.data.pdf_transformer import shift_boxes, spans_to_lines, lines_to_chunks
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead

#from src.data.utils_image import bimg2np
#from PIL import Image

def add_column(df, func, *names):
    output = func(df) if len(names)>1 else tuple([func(df)])
    for name, lines in zip(names, output):
        df[name] = lines
    return df

class PDFProcessor:
    def __init__(self, data_dir, **kwargs):
        super(PDFProcessor, self).__init__(**kwargs)
        self.df = None
        self.dir = os.path.join(*re.split(r'[\/\\]', data_dir))
        self.pdf_path = glob.glob(os.path.join(self.dir,'*.pdf'))[0]
        
        self.chunk_path = os.path.join(self.dir, 'chunks.tsv')
        self.line_path = os.path.join(self.dir, 'lines.tsv')
        self.span_path = os.path.join(self.dir, 'spans.tsv')
        
        self.bbox_path = os.path.join(self.dir, 'bboxes.pkl')
        self.image_path = os.path.join(self.dir, 'images.pkl')
        self.question_path = os.path.join(self.dir, 'questions.pkl')
        
        

    def extract_from_pdf(self, fix_boxes=True, verbose=False):
        span_df, lines_bboxes, images_bytes =  pdf_to_spans(self.pdf_path)
        
        line_df = spans_to_lines(span_df, lines_bboxes)
        line_df = self.featurize_lines(line_df, fix_boxes=fix_boxes, verbose=verbose)
        line_df.to_csv(self.line_path, sep='\t', index=False)
        span_df.to_csv(self.span_path, sep='\t', index=False)
        save_pickle(lines_bboxes, self.bbox_path)
        
        for k,v in images_bytes.items():
            image = Image.fromarray(bimg2np(v)).convert('RGB')
            image.save(os.path.join(self.dir, 'images', "{}.jpg".format(re.sub(r'[\<\>]', '', k))))

    def extract_chunks(self):
        if not os.path.exists(self.line_path):
            self.extract_from_pdf(fix_boxes=True, verbose=False)
            
        chunk_df = lines_to_chunks(pd.read_csv(self.line_path, sep='\t').fillna(''))
        chunk_df = self.featurize_chunks(chunk_df)
        chunk_df.to_csv(self.chunk_path, sep='\t', index=False)
            
    def featurize_lines(self, line_df, fix_boxes=False, verbose=False):
        if fix_boxes:
            line_df = shift_boxes(line_df, verbose=verbose)
            
        line_df = add_column(line_df, make_caption, 'caption')
        line_df = add_column(line_df, make_mask, 'mask')
        line_df = add_column(line_df, make_spacing_back, 'spacing')
        line_df = add_column(line_df, make_chunks, 'chunk', 'group', 'type')
        line_df = add_column(
            line_df, lambda x: make_sections(x, verbose=verbose), 
            'section', 'subsection', 'subsubsection', 'title', 'section_tag'
        )
        line_df = add_column(line_df, make_image_urls, 'image_url')
        return line_df
    
    def featurize_chunks(self, chunk_df):
        model_t5 = AutoModelWithLMHead.from_pretrained("t5-base")
        tokenizer_t5 = AutoTokenizer.from_pretrained("t5-base")
        chunk_df = make_chunk_mask(chunk_df)
        chunk_df = make_chunk_summaries(chunk_df, model_t5, tokenizer_t5)
        chunk_df = make_group_summaries(chunk_df, model_t5, tokenizer_t5)
        chunk_df = make_section_summaries(chunk_df, model_t5, tokenizer_t5)
        return chunk_df
    
    def load_spans(self):
        if not os.path.exists(self.span_path):
            self.extract_from_pdf(fix_boxes=True, verbose=False)
        self.df = pd.read_csv(self.span_path, sep='\t').fillna('')
        
    def load_lines(self):
        if not os.path.exists(self.line_path):
            self.extract_from_pdf(fix_boxes=True, verbose=False)
        self.df = pd.read_csv(self.line_path, sep='\t').fillna('')
        
    def load_chunks(self):
        if not os.path.exists(self.chunk_path):
            self.extract_chunks()
        self.df = pd.read_csv(self.chunk_path, sep='\t').fillna('')
