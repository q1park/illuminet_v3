import re
import numpy as np
from collections import Counter
from src.data.structures import LineData
from src.data.utils import reget, unique, box_edges, box_counter, box_counter_page

#####################################################################################
### Functions to make line spacing feature
#####################################################################################

def make_spacing_front(line_df):
    box = list(line_df[['x0', 'y0', 'x1', 'y1']].to_records(index=False))
    ytb = box_edges(box, 1, 3)
    return [np.around(ytb[i+1][0]-ytb[i][1], 1) for i in range(0, len(ytb)-1)]+[-99.]


def make_spacing_back(line_df):
    box = list(line_df[['x0', 'y0', 'x1', 'y1']].to_records(index=False))
    ytb = box_edges(box, 1, 3)
    return [-99.]+[np.around(ytb[i][0]-ytb[i-1][1], 1) for i in range(1, len(ytb))]


#####################################################################################
### Functions to make document indentation feature
#####################################################################################

def make_indent(line_df):
    box = list(line_df[['x0', 'y0', 'x1', 'y1']].to_records(index=False))
    laligns = box_edges(box, 0, rnd=1)
    counts = Counter(laligns)
    base = sorted([(k,v) for k,v in counts.items() if k<25], key=lambda x: x[1], reverse=True)[0][0]
    
    indent_tags = {base:0}
    idx = 1 # aggregating the tags for each font size
    idx_unk = 1

    for lalign in sorted(list(counts.keys())):
        if lalign>base:
            indent_tags[lalign] = -idx_unk
            idx_unk+=1
        if lalign<base:
            indent_tags[lalign] = idx
            idx += 1
    return [indent_tags[x] for x in laligns]

#####################################################################################
### Functions to make chunks, groups, and types
#####################################################################################

import string

BACK_SEP = 0.3
FRONT_SEP = 0.3

def make_alphanum_dict():
    alphanum_dict = {}
    alphanum_dict.update(dict(zip(
        list(string.ascii_lowercase), 
        [str(x) for x in range(len(string.ascii_lowercase))]
    )))
    alphanum_dict.update(dict(zip(
        list(string.ascii_uppercase), 
        [str(x) for x in range(len(string.ascii_uppercase))]
    )))
    return alphanum_dict

alphanum_dict = make_alphanum_dict()
re_strip = re.compile(r'^[^a-zA-Z0-9\•]|[^a-zA-Z0-9\•]$')

def make_tag(prefix):
    tag = re_strip.sub('', prefix)
    
    if tag.isalpha():
        return alphanum_dict[tag]
    else:
        return tag


def make_chunks(line_df):
    sfront = make_spacing_front(line_df)
    sback = make_spacing_back(line_df)
    
    is_list = False
    is_enum = False
    
    lists_enums = line_df[line_df['marker']>0]
    lists_enums_iloc = dict(zip(lists_enums.index, range(len(lists_enums.index))))
    lists_enums_loc = dict(zip(range(len(lists_enums.index)), lists_enums.index))
    
    chunks = []
    groups = []
    types = []
    n_chunk = -1
    n_group = 0
    
    for i, row in line_df.iterrows():
        f,b = sfront[i], sback[i]
        next_f = sfront[i+1] if i<len(sfront)-1 else -99.
        
        not_front_not_back = np.abs(f)>FRONT_SEP and np.abs(b)>BACK_SEP
        front_not_back = not np.abs(f)>FRONT_SEP and np.abs(b)>BACK_SEP
        not_front_back = np.abs(f)>FRONT_SEP and not np.abs(b)>BACK_SEP
        
        this_line, this_tag, this_marker = i, make_tag(row.prefix), row.marker
        
        if this_marker>0 and lists_enums_iloc[i]<len(lists_enums)-1:
            idx = lists_enums_iloc[i]
        
            next_line = lists_enums_loc[idx+1]
            next_tag = make_tag(lists_enums.iloc[idx+1].prefix)
            next_marker = lists_enums.iloc[idx+1].marker
        else:
            next_line = -1
            next_tag = ''
            next_marker = -1

        if this_marker==1:
            n_chunk+=1
            _type = 2
            if not is_list:
                is_list = True
                n_group+=1

            if (next_marker==1 and next_line-this_line<15):
                is_list = True
            else:
                is_list = False
        elif this_marker==2:
            _type = 3
            n_chunk+=1
            if not is_enum and next_tag.isdigit() and int(next_tag)==int(this_tag)+1:
                is_enum = True
                n_group+=1
            if next_tag.isdigit() and int(next_tag)==int(this_tag)+1:
                is_enum = True
            else:
                is_enum = False

        else:
            if is_list:
                _type = 2
            elif is_enum:
                _type = 3
            elif not_front_not_back:
                _type = 0
                if not re.search(r'^[a-z]', row.content):
                    n_chunk+=1
                    if len(types)>0 and types[-1] != _type:
                        n_group+=1
            elif front_not_back:
                _type = 1
                if not re.search(r'^[a-z]', row.content):
                    n_chunk+=1
                    if len(types)>0 and types[-1] != _type:
                        n_group+=1
            else:
                _type = 1

        chunks.append(n_chunk)
        groups.append(n_group)
        types.append(_type)
    assert len(chunks)==len(line_df)
    return chunks, groups, types

#####################################################################################
### Functions to make document mask feature
#####################################################################################

def group_df_dict(df, by, *args):
    if len(args)==0:
        args = tuple(df.columns)
    return {k:list(v[list(args)].to_records(index=False)) for k,v in df[[by]+list(args)].groupby(by)}

def get_trim_boxes(content, box, pad):
    length = len(content)
    head_counts = Counter()
    tail_counts = Counter()
    
    for (n,c), b in zip(content.items(), box.values()):
        head_feats = [_e for _e, _c in zip(box_edges(b[:pad],0,2), c[:pad]) if len(' '.join(_c)) < 50]
        tail_feats = [_e for _e, _c in zip(box_edges(b[-pad:],0,2), c[-pad:]) if len(' '.join(_c)) < 50]
        head_counts.update(head_feats)
        tail_counts.update(tail_feats)
        
    return np.array(
        [b for b,n in head_counts.most_common(20) if n/length>0.2]
        +[b for b,n in tail_counts.most_common(20) if n/length>0.2]
    )

def make_page_mask(content, box, pad=3):
    page_mask = {n:[] for n in content.keys()}
    artifacts = get_trim_boxes(content, box, pad)
    regex = []
    
    def from_edges(n, idx, line, edges):
        nz = np.around(np.sum(np.abs(artifacts-np.array(edges)), axis=1), 5)
        if (len(np.argwhere(nz==0.))>0 and len(' '.join(line))<50) or (len(line)==1 and line[0].isdigit()):
            page_mask[n].append(idx)
            if (
                len(regex)<=10 and line[0][:20].strip() not in regex and 
                re.search(r'[a-zA-Z]', line[0][:20].strip())
            ):
                regex.append(line[0][:20].strip())
    
    for (n,c), b in zip(content.items(), box.values()):
        for idx, (edges, _c) in enumerate(zip(box_edges(b[:pad], 0, 2), c[:pad])):
            from_edges(n, idx, _c, edges)
        for idx, (edges, _c) in enumerate(zip(box_edges(b[-pad:], 0, 2), c[-pad:])):
            from_edges(n, len(c)-pad+idx, _c, edges)

    def from_regex(n, idx, line):
        if len(regex)>0 and re.search(r'|'.join(regex), line[0][:20]):
            page_mask[n].append(idx)
            
    for n,c in content.items():
        for idx, _c in enumerate(c[:pad]):
            from_regex(n, idx, _c)
        for idx, _c in enumerate(c[-pad:]):
            from_regex(n, len(c)-pad+idx, _c)
    return page_mask

def make_mask(line_df):
    content = group_df_dict(line_df, 'page', 'content')
    box = group_df_dict(line_df, 'page', 'x0', 'y0', 'x1', 'y1')
    page_mask = make_page_mask(content, box, pad=3)
    doc_mask = []

    for page,boxes in box.items():
        for i,_ in enumerate(boxes):
            if i in page_mask[page] or content[page][i][0].startswith('<i'):
                doc_mask.append(1)
            else:
                doc_mask.append(0)

    assert len(doc_mask)==len(line_df)
    return doc_mask

#####################################################################################
### Functions to make table of content feature
#####################################################################################

def find_toc_candidates(boxes, mask, window_size=10, right_margin=75, min_width=50, rnd=1, verbose=False):
    windows = []
    l, r = 0.0, 0.0
    idx_start = 0
    start = False
    
    for idx in range(len(boxes)):
        win = boxes[idx:idx+window_size]
        win_mask = mask[idx:idx+window_size]
        
        counter_r = Counter([x[0] for x in zip(box_edges(win, 2, rnd=rnd), win_mask) if x[1]==0])
        counter_lr = Counter([x[0] for x in zip(box_edges(win, 0, 2, rnd=rnd), win_mask) if x[1]==0])
        
        counts_r = max(counter_r[np.around(r+x, rnd)] for x in np.arange(-0.2,0.3,0.1))
        (most_l, most_r), counts_most = counter_lr.most_common(1)[0]

        if not start and counts_most/len(counter_r)>0.8:
            if verbose:
                print('starting at', idx)
            
            l, r = most_l, most_r
            idx_start = idx
            start = True
            
        elif start and counts_r/len(counter_r)==0.0:
            if r>right_margin and (r-l)>min_width:
                windows.append((r, (idx_start, idx)))
                
                if verbose:
                    print('ending at', idx, (l, r))
            else:
                if verbose:
                    print('ignoring section', idx, (l, r))
                    
            l, r = 0.0, 0.0
            start = False
            
    if verbose:
        print('possible tocs:', windows)
    return windows

def find_toc(line_df, window_size=10, right_margin=75, min_width=50, rnd=1, verbose=False):
    toc_cand = find_toc_candidates(
        list(line_df[['x0', 'y0', 'x1', 'y1']].to_records(index=False)), 
        list(x[0] for x in line_df[['mask']].to_records(index=False)), 
        window_size=window_size, 
        right_margin=right_margin, 
        min_width=min_width, 
        rnd=rnd, 
        verbose=verbose
    )
    
    if verbose:
        print('choosing first candidate: ', toc_cand[0])
    
    _, (toc_start, toc_end) = toc_cand[0]

    for n, seg in tuple(map(lambda x: (x[0],(x[1].index[0], x[1].index[-1]+1)), line_df.groupby('page'))):
        idx_start, idx_end = seg
        if toc_start>idx_start and toc_start<idx_end:
            if np.abs(toc_start-idx_end)<4:
                toc_start = idx_end
            else:
                toc_start = idx_start
            break
    return toc_start, toc_end

def toc_structure(toc_df, verbose=False):
    def substructure(boxes):
        left_edges = box_edges(boxes, 0, rnd=1)
        counts = Counter(left_edges)

        base = min([x[0] for x in counts.items() if x[1]>1])

        indent_tags = {}
        idx = 0 # aggregating the tags for each font size
        idx_unk = 1

        for l in sorted(list(counts.keys())):
            if l<base or l>25:
                indent_tags[l] = -idx_unk
                idx_unk+=1
            else:
                indent_tags[l] = idx
                idx += 1
        return [indent_tags[x] for x in left_edges]
    
    _title_page = lambda x: (reget(r'[a-zA-Z\(\)].+?(?=([\.\_])\1+)|[a-zA-Z\(\)].+$', x), reget(r'[0-9]+$', x))

    content = list(toc_df[['content']].to_records(index=False))
    size = list(x for x in toc_df[['size']].to_records(index=False))
    face = list(x for x in toc_df[['face']].to_records(index=False))
    box = list(toc_df[['x0', 'y0', 'x1', 'y1']].to_records(index=False))
    spacing = list(x[0] for x in toc_df[['spacing']].to_records(index=False))
    mask = list(x[0] for x in toc_df[['mask']].to_records(index=False))
    
    content = [list(_title_page(' '.join(c))) for c in content]
    spacing = [-99.]+spacing[1:]
    levels = substructure(box)
    
    numtags = set([x for x,(_,page) in zip(levels, content) if x>=0 if page is not None])
    common_space = Counter(spacing).most_common()[0][0]
    
    _lvl, _space = [], []
    _c, _s, _f = [], [], []
    
    for i in range(len(levels)):
        if mask[i]==1:
            continue
        if levels[i]>=0 and content[i][0]:
            if len(_c)>0 and spacing[i]-common_space<0.2 and (
                (content[i][-1] is not None and _c[-1][-1] is None and _lvl[-1] in numtags) or 
                (content[i][-1] is None and _c[-1][-1] is None 
                 and size[i][0]==_s[-1][-1] and face[i][0]==_f[-1][-1])
            ):
                _c[-1][-1] = content[i][-1]
                _c[-1][0] += ' '+content[i][0]
                if verbose:
                    print('fixed:', _c[-1][0]+' || '+content[i][0])
            else:
                _lvl.append(levels[i])
                _c.append(content[i])
                _s.append(size[i])
                _f.append(face[i])
                _space.append(spacing[i])
    toc = []
    state = [0]*len(set(_lvl))
    
    for lvl, line in zip(_lvl, _c):
        if re.search(r'^table|contents$', line[0].lower()):
            continue
        elif sum(state)==0 or lvl>=last_lvl:
            state[lvl]+=1
        else:
            state[lvl]+=1
            for l in range(lvl+1, len(state)):
                state[l]=0
        title = line[0].strip() if line[0] else line[0]
        page = int(line[1]) if line[1] else line[1]
        toc.append((tuple(state), title, page))
        last_lvl = lvl
    return toc

def make_toc(line_df, verbose=False):
    toc_start, toc_end = find_toc(line_df, verbose=verbose)
    toc_df = line_df.iloc[toc_start:toc_end]
    toc = toc_structure(toc_df, verbose=verbose)
    
    toc_data = [x[1] for x in toc]
    toc_index_map = {x[0]:i for i,x in enumerate(toc)}
    toc_lines = [None]*toc_start+[1]*(toc_end-toc_start)+[None]*(len(line_df)-toc_end)
    
    
    assert len(toc_lines)==len(line_df)
    return LineData(name='toc', lines=toc_lines, data=toc_data, index_map = toc_index_map, segment=True)

#####################################################################################
### Functions to make document header candidates
#####################################################################################

def longest_word(string):
    string = re.sub(r'[^a-zA-Z\&\s]', ' ', re.sub(r'\/', ' ', string))
    words = [x for x in string.split() if len(x)>0]
    sorted_by_len = list(sorted(zip(words, [len(x) for x in words]), key=lambda x: x[1], reverse=True))
    return sorted_by_len[0][0]

def make_header_candidates(line_df, toc_data):
    sfront = make_spacing_front(line_df)
    sback = make_spacing_back(line_df)
    
    common_space = Counter(line_df.spacing.tolist()).most_common()[0][0]
    key_words = r'|'.join([re.escape(longest_word(x)) for x in toc_data.data])
    max_len = max([len(x[0]) for x in toc_data.data])
    
    header_idx = 0
    header_lines = []
    header_map = {}
    prev_indents = 0
    prev_face = 0

    for idx, row in line_df.iterrows():
        prev_type = line_df.iloc[idx-1].types if idx>0 else 0
        text = re.sub(r'\s+', ' ', row.c0).strip()

        if (
            re.search(r'^[a-z]', text) is not None or
            row['mask']>0 or 
            row['types']>1 or 
            (np.abs(sfront[idx])<0.4 and np.abs(sback[idx])<0.4) or
            re.search(key_words, text) is None or
            toc_data.lines[idx] is not None
        ):
            header_lines.append(None)
        else:
            header_lines.append(header_idx)
            header_map[header_idx] = text
            header_idx+=1

    header_data = list(header_map.values())
    header_index_map = {k:i for i,k in enumerate(header_map.keys())}
    return LineData(name='header', lines=header_lines, data=header_data, index_map=header_index_map, segment=True)

#####################################################################################
### Function to "properly" classify section boundaries
#####################################################################################

from src.nlp.tfidf import TFIDFVectors

def get_search_matrices(match_path, len_lines):
    segments = []
    idx_toc_start, idx_header_start = 0, 0
    start = False
    
    for k,v in match_path.items():
        if not start and v is None:
            idx_toc_start = k
            idx_header_start = match_path[k-1] if k>0 else 0
            start = True
        elif start and v is not None:
            segments.append(((idx_toc_start, k), (idx_header_start, v)))
            start = False
            
    if start:
        segments.append(((idx_toc_start, k+1), (idx_header_start, len_lines)))
        start = False
            
    return segments

def filter_matches(match_idxs):
    matches = []
    header_idx = -1

    for toc_idx in np.unique(match_idxs[:,0]):
        toc_matches = match_idxs[(match_idxs[:,0]==toc_idx)&(match_idxs[:,1]>header_idx)][:,1]

        if len(toc_matches)>0:
            header_idx = np.min(toc_matches)
            matches.append((toc_idx, header_idx))
    return matches

def update_match_path(match_path, sim, threshold, len_lines):
    for toc, head in get_search_matrices(match_path, len_lines):
        raw_idxs = filter_matches(np.argwhere(sim[slice(*toc), slice(*head)]>threshold))
        match_idxs = [(x+toc[0],y+head[0]) for x,y in raw_idxs]
        
        for k,v in match_idxs:
            match_path[k] = v
    return match_path

def match_toc(toc, header, verbose=False):
    len_lines = len(header.lines)
    tfidf = TFIDFVectors(*toc.data)
    sim = (tfidf.vectors@tfidf.query(*header.data).T).toarray()
    
    match_path = dict([(i, None) for i in range(len(sim))])
    
    for thresh in [0.99, 0.95, 0.85, 0.5]:
        match_path = update_match_path(match_path, sim, threshold=thresh, len_lines=len_lines)
        
    if verbose:
        for k,v in match_path.items():
            if v is None:
                print('not found', toc.data[k])
            else:
                print('found', toc.data[k], ' || ', header.data[v])

    return {k:header.segments[v][0][0] if v is not None else None for k,v in match_path.items()}


def make_section_segments(line_df, verbose):
    def next_notnull(match_path, idx):
        for i in range(idx, list(match_path.keys())[-1]+1):
            if match_path[i] is not None:
                return match_path[i]
    toc_data = make_toc(line_df)
    header_data = make_header_candidates(line_df, toc_data)
    
    match_path = match_toc(toc_data, header_data, verbose=verbose)

    toc_start, toc_end = toc_data.segments[1][0]
    sections = {(-1,-1,-1):['TOC', toc_start, toc_end], (0,0,0):['START', 0, None]}

    for i,(k,v) in enumerate(toc_data.index_map.items()):
        if len(k)<3:
            k+=(0,)
        elif len(k)>3:
            k = k[:3]
        start = match_path[v] if match_path[v] is not None else next_notnull(match_path, v)
        sections[k]=[toc_data.data[v], start, None]
    sections = dict(sorted(sections.items(), key=lambda x: x[1][1]))
    sections_keys = list(sections.keys())

    for i in range(len(sections_keys)):
        this_section_start = sections[sections_keys[i]][1]
        next_section_start = sections[sections_keys[i+1]][1] if i<len(sections_keys)-1 else len(toc_data.lines)

        sections[sections_keys[i]][2]=next_section_start
    return {k:tuple(v) for k,v in sections.items()}

def make_sections(line_df, verbose):
    section_segments = make_section_segments(line_df, verbose=verbose)
        
    sections, subsections, subsubsections = [], [], []
    
    for (s,sb,sbsb), (name, i, f) in section_segments.items():
        sections.extend([s]*(f-i))
        subsections.extend([sb]*(f-i))
        subsubsections.extend([sbsb]*(f-i))
    return sections, subsections, subsubsections