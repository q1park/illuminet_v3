import re
import fitz
import pandas as pd

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
    0:re.compile(r'^[0-9]{1,2}([\.\-][0-9]{1,2})+[\.\-]{0,1}(?=$)|^[0-9]{1,2}([\.\-][0-9]{1,2})+[\.\-]{0,1}(?=\s[A-Z])'),
    1:re.compile(r'^[\•\-](?=$|\s)'),
    2:re.compile(r'^[\(]{0,1}[0-9]{1,2}[\.\-\)]{0,1}(?=$|\s)|^[\(]{0,1}[a-kA-K][\.\-\)](?=$|\s)')
}

def split_markers(text):
    for tag,rx in re_markers.items():
        search = rx.search(text)
        match = search.group() if search else None
        if match and not (match.strip().isdigit() and int(match.strip())>100):
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

def pdf_to_spans(input_path):
    def scale_box(box, h, w):
        norms = [100./x for x in [w,h,w,h]]
        return tuple(b*norm for b,norm in zip(box, norms))
    
    span_dict = {k:[] for k in ['span', 'size', 'face', 'marker', 'line', 'page']}
    lines_bboxes = {}
    images_bytes = {}
    
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
                            lines_bboxes[n_line] = (x0_last, y0, x1, y1)
                            x0_last, y0_last, x1_last, y1_last = lines_bboxes[n_line]
                        else:
                            n_line+=1
                            lines_bboxes[n_line] = x0, y0, x1, y1
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
                
                lines_bboxes[n_line] = scale_box(b['bbox'], height, width)
                images_bytes[img_label] = b['image']

    assert all(len(span_dict['span'])==len(span_dict[x]) for x in ['size', 'face', 'marker', 'line', 'page'])
    return pd.DataFrame(span_dict), lines_bboxes, images_bytes