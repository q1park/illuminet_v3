import re
import copy
import numpy as np
import pandas as pd
from collections import Counter
from src.data.utils import group_df_dict, jaccard_matrix, box_edges, box_counter, box_counter_page

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

def shift_boxes(line_df, verbose=False):
    box = group_df_dict(line_df, 'page', *['x0', 'y0', 'x1', 'y1'])
    
    box_lines = [x for n in box.values() for x in n]
    box = shift_page(box, topk=10, pad=None, verbose=verbose)
    box = shift_page(box, topk=10, pad=3, force=False, verbose=verbose)
    new_box_lines = [x for n in box.values() for x in n]
    assert len(new_box_lines)==len(box_lines)
    
    line_df.update(pd.DataFrame([list(x) for x in new_box_lines], columns = ['x0', 'y0', 'x1', 'y1']))
    return line_df

#####################################################################################
### Functions make lines dataframe
#####################################################################################

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

def spans_to_lines(span_df, lines_bboxes):
    line_df = pd.DataFrame(
        list(map(lambda x: flatten_line(x[1]), span_df.groupby('line'))),
        columns = ['prefix', 'c0', 'content', 'size', 'face', 'marker', 'line', 'page']
    )

    box_df = pd.DataFrame(list(lines_bboxes.values()), columns = ['x0', 'y0', 'x1', 'y1'])
    return pd.concat([line_df, box_df], axis=1)

#####################################################################################
### Functions make chunks dataframe
#####################################################################################

def first_nonempty(df, key):
    nonempty = df[df[key]!=''][key]
    return nonempty.iloc[0] if len(nonempty)>0 else ''

def flatten_chunk(chunk_df):
    prefix = first_nonempty(chunk_df, 'prefix')
    content = ' '.join(chunk_df['content']).strip()
    marker = chunk_df['marker'].iloc[0]
    chunk = chunk_df['chunk'].iloc[0]
    group = chunk_df['group'].iloc[0]
    section = chunk_df['section'].iloc[0]
    subsection = chunk_df['subsection'].iloc[0]
    subsubsection = chunk_df['subsubsection'].iloc[0]
    title = first_nonempty(chunk_df, 'title')
    section_tag = chunk_df['section_tag'].iloc[0]
    image_url = first_nonempty(chunk_df, 'image_url')
    return prefix, content, marker, chunk, group, section, subsection, subsubsection, title, section_tag, image_url
    

def lines_to_chunks(line_df):
    dropcols = [
        'c0', 'size', 'face', 'line', 'page', 'caption', 'mask', 
        'x0', 'y0', 'x1', 'y1', 'spacing', 'type'
    ]
    newcols = [
        'prefix', 'content', 'marker', 'chunk', 'group',
        'section', 'subsection', 'subsubsection', 'title', 'section_tag', 'image_url'
    ]

    clean_df = line_df[(line_df['mask']==0)&(line_df['section']>=0)&(line_df['subsection']>=0)]
    clean_df = clean_df.drop(columns=dropcols).reset_index(drop=True)
    
    chunks_idx, groups_idx = 0, 0
    chunks_new, groups_new = [], []
    
    for i in range(0, len(clean_df)-1):
        this_row, next_row = clean_df.iloc[i], clean_df.iloc[i+1]
        chunks_new.append(chunks_idx)
        groups_new.append(groups_idx)
        
        if re.search(r'^[a-z]', next_row['content']):
            pass
        else:
            if next_row['chunk']!=this_row['chunk'] or this_row['section_tag']!=next_row['section_tag']:
                chunks_idx+=1
                
                if next_row['group']!=this_row['group'] or next_row['marker']==0 or this_row['section_tag']!=next_row['section_tag']:
                    groups_idx+=1
    chunks_new.append(chunks_idx)
    groups_new.append(groups_idx)
    
    clean_df['group'] = groups_new
    clean_df['chunk'] = chunks_new

    chunk_df = pd.DataFrame(
        list(map(lambda x: flatten_chunk(x[1]), clean_df.groupby('chunk'))),
        columns = newcols
    )
    return chunk_df.drop(columns=['chunk'])

