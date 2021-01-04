import os
import re
import copy
from src.data.structures import LineData
from lxml import etree as ET

#####################################################################################
### Functions to extract structure from (T)able (O)f (C)ontents xml
#####################################################################################

def get_structure_list(unit, flat=[], depth=0, struct=None):
    unit_dict = {}
    depth+=1
    
    if struct is None:
        struct = []
        flat = []
        
    if len(struct)<=depth-1:
        struct+=[0]
    
    for u in unit.xpath('Unit'):
        utype = u.attrib['UnitType']
        uname, uid, upage = tuple(u.find(x).text for x in ['UnitName', 'UnitID', 'PageID'])
        
        _depth = depth
        struct[depth-1]+=1
        flat.append({
            'struct':copy.deepcopy(struct), 
            'depth':depth, 
            'uid':uid, 
            'type':utype, 
            'name':uname, 
            'pageid':upage
        })
        
        if len(u.xpath('Unit'))>0:
            assert upage is None
            flat = get_structure_list(u, flat, depth, struct)
            
            for i in range(depth-1+1, len(struct)):
                struct[i] = 0
                
            depth = _depth
    return flat


def get_structure_dict(unit, depth=0):
    unit_dict = {}
    depth+=1
    
    for u in unit.xpath('Unit'):
        utype = u.attrib['UnitType']
        uname = u.find('UnitName').text
        uid = u.find('UnitID').text
        upage = u.find('PageID').text
        
        if len(u.xpath('Unit'))>0:
            assert upage is None
            _depth = depth
            unit_dict[uid] = {
                'type':utype, 
                'name':uname, 
                'depth':depth, 
                'sub':get_structure_dict(u, depth)
            }
            
            depth = _depth
        else:            
            unit_dict[uid] = {
                'type':utype, 
                'name':uname, 
                'depth':depth, 
                'pageid':upage
            }
    return unit_dict

def get_structure(path):
    tree = ET.parse(path)
    first_unit = tree.find('.//Unit')
    structure = get_structure_list(first_unit)
    
    max_depth = len(structure[-1]['struct'])
    
    for page in structure:
        if len(page['struct'])<max_depth:
            page['struct'] = tuple(page['struct']+[0]*(max_depth-len(page['struct'])))
        else:
            page['struct'] = tuple(page['struct'])
    return structure

#####################################################################################
### Functions to extract text data and multimedia urls from page xml
#####################################################################################

## Assumptions: all valuable content within ContentWindow
content_base = 'ContentWindow'

## Assumptions: valuable content limited to TextDisplay, Image, Video
content_tags = {
    'TextDisplay':['Name', 'MarkupText'], 
    'Image':['Name', '_URL'], 
    'Video':['Name', '_URL']
}

def get_page_content(path):
    tree = ET.parse(path)
    content_dict = {}
    content_list = tree.xpath('|'.join(['.//{}/{}'.format(content_base, c) for c in content_tags.keys()]))
    
    for i, content in enumerate(content_list):
        label = '{}_{}'.format(i, content.tag)
        content_dict[label] = {}
        
        for field in content_tags[content.tag]:
            ## Assumptions: each content element has only one value
            data = content.find('{}/Value'.format(field)).text
            content_dict[label][field] = data
    return content_dict

#####################################################################################
### Functions to parse markup text
#####################################################################################

def xml_lines(markup_raw):
    return [
        re.sub(r'\&nbsp\;', '', y) .strip()
        for x in re.findall(r'(?<=\<p\>).+?(?=\<\/p\>)|(?<=\<\/p\>)\<[^p].+?(?=\<p\>)', markup_raw)
        for y in re.split(r'\<br\>|(?<=.)\&nbsp\;$', x) 
        if len(y)>0 
        and re.search(r'^\&nbsp\;$', y) is None
    ]

def xml_lists(markup_list):
    listed_text = []
    listed_labels = []
    
    for text in markup_list:
        if re.search(r'^\<ol\>.+\<\/ol\>$', text):
            listed_list = [x for x in re.findall(r'(?<=\<li\>).+?(?=\<\/li\>)', text)]
            listed_text.extend(listed_list)
            listed_labels.extend([2]*len(listed_list))
        elif re.search(r'^\<ul\>.+\<\/ul\>$', text):
            listed_list = [x for x in re.findall(r'(?<=\<li\>).+?(?=\<\/li\>)', text)]
            listed_text.extend(listed_list)
            listed_labels.extend([1]*len(listed_list))
        else:
            listed_text.append(text)
            listed_labels.append(None)
    return listed_text, listed_labels

#####################################################################################
### Functions to extract dif data to RawData and Features objects
#####################################################################################

def dif_extract(dir_dir, page_dir):
    struct = get_structure(os.path.join(dir_dir, 'CourseTOC.xml'))
    
    structure = {'name':'structure', 'lines':[], 'data':struct}
    content = {'name':'content', 'lines':[]}
    name = {'name':'name', 'lines':[]}
    mtype = {'name':'type', 'lines':[]}
    image = {'name':'image', 'lines':[], 'data':[], 'index_map':{}}
    video = {'name':'video', 'lines':[], 'data':[], 'index_map':{}}

    
    raw = {k:[] for k in ['content', 'name', 'type']}
    ol_lines = []
    idx_img, idx_vid = 0, 0
    inv_img_map, inv_vid_map = {}, {}
    
    for page in struct:
        if page['pageid'] is None:
            content['lines'].append(page['name'])
            name['lines'].append(page['name'])
            mtype['lines'].append(page['type'])
        else:
            page_file = os.path.join(page_dir, 'Page_{}.xml'.format(page['pageid']))
            for k,v in get_page_content(page_file).items():
                if k.endswith('TextDisplay'):
                    text, ol = xml_lists(xml_lines(v['MarkupText']))
                    content['lines'].extend(text)
                    name['lines'].extend([page['name']]*len(text))
                    mtype['lines'].extend([page['type']]*len(text))
                    ol_lines.extend(ol)

                elif k.endswith('Image'):
                    if v['_URL'] not in inv_img_map:
                        inv_img_map[v['_URL']] = '<img{}>'.format(str(idx_img).zfill(3))
                        idx_img+=1
                        
                    content['lines'].append(inv_img_map[v['_URL']])
                    name['lines'].append(page['name'])
                    mtype['lines'].append(page['type'])
                    ol_lines.append(None)
                        
                elif k.endswith('Video'):
                    if v['_URL'] not in inv_vid_map:
                        inv_vid_map[v['_URL']] = '<vid{}>'.format(str(idx_vid).zfill(3))
                        idx_vid+=1
                        
                    content['lines'].append(inv_vid_map[v['_URL']])
                    name['lines'].append(page['name'])
                    mtype['lines'].append(page['type'])
                    ol_lines.append(None)
    
    image['lines'] = [x if re.search(r'^\<img', x) else None for x in content['lines']]
    video['lines'] = [x if re.search(r'^\<vid', x) else None for x in content['lines']]
#     img_map = {v:k for k,v in inv_img_map.items()}
#     vid_map = {v:k for k,v in inv_vid_map.items()}
    
    image['data'] = list(inv_img_map.keys())
    image['index_map'] = dict(zip(inv_img_map.values(), range(len(image['data']))))
    video['data'] = list(inv_vid_map.keys())
    video['index_map'] = dict(zip(inv_vid_map.values(), range(len(video['data']))))
    
    return structure, content, name, mtype, image, video
#####################################################################################
### 
#####################################################################################