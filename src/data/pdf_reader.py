import re
import copy
import fitz
import numpy as np
import pandas as pd
from collections import Counter
from src.data.utils import unique, jaccard_matrix, box_edges, box_counter, box_counter_page

#####################################################################################
### Functions to extract raw data from pdf
#####################################################################################

def read_flags(flags):
    l = []
    if flags & 2**4:
        l.append("bold")
    if flags & 2**1:
        l.append("italic")
    if flags & 2**0:
        l.append("superscript")
    return l

re_markers = {
    0:re.compile(r'^[a-zA-Z0-9]{1,2}([^a-zA-Z0-9][a-zA-Z0-9]{1,2})+.{0,1}(?=$|\s)'),
    1:re.compile(r'^\•(?=$|\s)'),
    2:re.compile(r'^[a-zA-Z0-9][^a-zA-Z0-9](?=$|\s)') #|^[a-zA-Z0-9](?=$)')
}

def split_markers(text):
    for tag,rx in re_markers.items():
        search = rx.search(text)
        match = search.group() if search else None
        if match and not (match.strip().isdigit() and int(match.strip())>10):
            splits = [match.strip(), re.sub(re.escape(match), '', text).strip()]
            splits = [x for x in splits if len(x)>0]
            return splits, [tag]+[-1]*(len(splits)-1)
    return [text], [-1]

def split_sentences(spans, markers):
    new_spans = []
    new_markers = []
    for span, marker in zip(spans, markers):
        sentences = re.split(r'(?<=[a-zA-Z0-9][\.\?\!])\s(?=[A-Z0-9])', span)
        sentences = [x.strip() for x in sentences if len(x.strip())>0]
        new_spans.extend(sentences)
        new_markers.extend([marker]+[-1]*(len(sentences)-1))
    return new_spans, new_markers

def format_line(line):
    spans, sizes, faces, markers = [], [], [], []
        
    for i, s in enumerate(line["spans"]):  # iterate through the text spans
        s_text = s['text'].strip()
        s_size = s['size']
        s_face = 1 if 'bold' in read_flags(s['flags']) else 0
        
        if len(s_text)>0:
            if i==0:
                span_list, marker_list = split_sentences(*split_markers(s_text))
            else:
                span_list, marker_list = split_sentences([s_text], [-1])
                
            spans.extend(span_list)
            sizes.extend([s_size]*len(span_list))
            faces.extend([s_face]*len(span_list))
            markers.extend(marker_list)

    assert all(len(spans)==len(x) for x in [sizes, faces, markers])
    return tuple(spans), tuple(sizes), tuple(faces), tuple(markers)

def flatten_line(line_df):
    size = line_df['size'].max()
    face = line_df['face'].max()
    line = line_df['line'].iloc[0]
    page = line_df['page'].iloc[0]
    marker = line_df['marker'].iloc[0]
    
    if marker>=0:
        content_prefix = line_df['span'].iloc[0]
        c0 = line_df['span'].iloc[1] if len(line_df['span'])>1 else ''
        content = ' '.join(line_df['span'].iloc[1:]).strip()
    else:
        content_prefix = ''
        c0 = line_df['span'].iloc[0]
        content = ' '.join(line_df['span']).strip()
    return content_prefix, c0, content, size, face, marker, line, page

def pdf_to_spans(input_path):
    def scale_box(box, h, w):
        norms = [100./x for x in [w,h,w,h]]
        return tuple(b*norm for b,norm in zip(box, norms))
    
    span_dict = {k:[] for k in ['span', 'size', 'face', 'marker', 'line', 'page']}
    lines_boxes = {}
    images_data = {}
    
    doc = fitz.open(input_path)
    img_idx = -1
    n_line = -1

    for i, p in enumerate(doc, start=1):
        height = p.rect[3]-p.rect[1]
        width = p.rect[2]-p.rect[0]
        x0_last, y0_last, x1_last, y1_last = (-99,)*4
        
        for b in p.getText("dict")["blocks"]:  # iterate through the text blocks
            if b['type'] == 0:  # this block contains text
                for line in b["lines"]:  # iterate through the text lines
                    spans, sizes, faces, markers = format_line(line)
                    x0, y0, x1, y1 = scale_box(line['bbox'], height, width)

                    if len(spans)>0:
                        if y0==y0_last and y1==y1_last and span_dict['size'][-1]>0:
                            lines_boxes[n_line] = (x0_last, y0, x1, y1)
                            x0_last, y0_last, x1_last, y1_last = lines_boxes[n_line]
                        else:
                            n_line+=1
                            lines_boxes[n_line] = x0, y0, x1, y1
                            x0_last, y0_last, x1_last, y1_last = x0, y0, x1, y1
                            
                        lines = [n_line]*len(spans)
                        pages = [i]*len(spans)

                        span_dict['span'].extend(spans)
                        span_dict['size'].extend(sizes)
                        span_dict['face'].extend(faces)
                        span_dict['marker'].extend(markers)
                        span_dict['line'].extend(lines)
                        span_dict['page'].extend(pages)

            elif b['type'] == 1:  # this block contains image
                n_line+=1
                img_idx+=1

                img_label = '<img{}>'.format(str(img_idx).zfill(4))
                size, face, marker = -1, -1, -1
                
                span_dict['span'].append(img_label)
                span_dict['size'].append(size)
                span_dict['face'].append(face)
                span_dict['marker'].append(marker)
                span_dict['line'].append(n_line)
                span_dict['page'].append(i)
                
                lines_boxes[n_line] = scale_box(b['bbox'], height, width)
                images_data[img_label] = b['image']

    assert all(len(span_dict['span'])==len(span_dict[x]) for x in ['size', 'face', 'marker', 'line', 'page'])
    return pd.DataFrame(span_dict), lines_boxes, images_data

def spans_to_lines(span_df, lines_boxes):
    line_df = pd.DataFrame(
        list(map(lambda x: flatten_line(x[1]), span_df.groupby('line'))),
        columns = ['prefix', 'c0', 'content', 'size', 'face', 'marker', 'line', 'page']
    )

    box_df = pd.DataFrame(list(lines_boxes.values()), columns = ['x0', 'y0', 'x1', 'y1'])
    return pd.concat([line_df, box_df], axis=1)

#####################################################################################
### Functions to correct for indentation shifts
#####################################################################################

def compute_shift(box, topk, verbose=False):
    page_counts, all_counts = box_counter_page(box, 0, rnd=3), box_counter(box, 0, rnd=3)
    topk_pages = {}

    for ralign,count in all_counts.most_common()[:topk]:
        topk_pages[ralign] = []
        for n in page_counts.keys():

            if ralign in page_counts[n]:
                topk_pages[ralign].append(n)
    
    jaccards, rows = jaccard_matrix(topk_pages, rnd=1) 
    idxs = {v:i for i,v in enumerate(rows)}
    
    for idx, in np.argwhere(jaccards.sum(axis=0)==1.):
        jaccards[idx,:] = np.ones(len(jaccards))
        jaccards[:,idx] = np.ones(len(jaccards)).T
    if verbose:
        print(jaccards)
    khi, klo = rows[np.argwhere(jaccards==0.)[0]]
    shift = khi-klo
    
    pages_to_shift = set()
    
    for i, k in enumerate(rows):
        if jaccards[idxs[khi], i]>0 and jaccards[idxs[klo], i]==0:
            pages_to_shift.update([
                n for n in topk_pages[k] 
                if n not in topk_pages[klo] and 
                min(page_counts[n].keys()) > klo
            ])
    if verbose:
        print(jaccards)
        print(khi, klo)

    return shift, pages_to_shift

def shift_page(box, topk=10, pad=None, force=False, verbose=False):
    _box = {k:box[k][:pad]+box[k][-pad:] for k in box} if pad else box
    shift, _to_shift = compute_shift(_box, topk=topk, verbose=verbose)
    
    tmp_box = {
        n:[(x[0]-shift,x[1],x[2]-shift,x[3]) for x in b] 
        if n in _to_shift else copy.deepcopy(b)
        for n,b in box.items()
    }
    
    ntopk = sum(x[1] for x in box_counter(box, 0, rnd=3).most_common(10))
    tmp_ntopk = sum(x[1] for x in box_counter(tmp_box, 0, rnd=3).most_common(10))
    
    if force:
        return tmp_box
    
    elif tmp_ntopk/ntopk>1.1:
        print('count ratio is {}.. shifting box'.format(tmp_ntopk/ntopk))
        return tmp_box
    
    else:
        print('count ratio is {}.. returning original box'.format(tmp_ntopk/ntopk))
        return box

def shift_boxes(box, verbose=False):
    box_lines = [x for n in box.values() for x in n]
    box = shift_page(box, topk=10, pad=None, verbose=verbose)
    box = shift_page(box, topk=10, pad=3, force=False, verbose=verbose)
    new_box_lines = [x for n in box.values() for x in n]
    assert len(new_box_lines)==len(box_lines)
    return dict(zip(range(len(new_box_lines)), new_box_lines))