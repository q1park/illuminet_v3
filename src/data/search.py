import numpy as np
import torch

from sklearn.preprocessing import normalize
from src.nlp.tfidf import CountVectors, TFIDFVectors

def make_digest_dict(chunk_df):   
    digest_dict = {}

    for k,v in chunk_df.groupby('section'):
        digest_dict[k] = {}

        for kk,vv in v.groupby('subsection'):
            digest_dict[k][kk] = {}

            for kkk,vvv in vv.groupby('subsubsection'):
                if vvv.iloc[0]['title']=='':
                    pass;
#                     raise ValueError('Section Error')
                
                digest_dict[k][kk][kkk] = {
                    'name':vvv.iloc[0]['title'], 
                    'content':vvv['section_summary']}
    return digest_dict

def make_cliffs_dict(chunk_df):   
    cliffs_dict = {}

    for k,v in chunk_df.groupby('section'):
        cliffs_dict[k] = {}

        for kk,vv in v.groupby('subsection'):
            cliffs_dict[k][kk] = {}

            for kkk,vvv in vv.groupby('subsubsection'):
                if vvv.iloc[0]['title']=='':
                    pass;
#                     raise ValueError('Section Error')
                
                cliffs_dict[k][kk][kkk] = {
                    'name':vvv.iloc[0]['title'], 
                    'content':[x for x in vvv['group_summary'].tolist() if len(x)>0]}
                    
    return cliffs_dict

def make_section_dict(chunk_df):   
    section_dict = {}

    for k,v in chunk_df.groupby('section'):
        section_dict[k] = {}

        for kk,vv in v.groupby('subsection'):
            section_dict[k][kk] = {}

            for kkk,vvv in vv.groupby('subsubsection'):
                if vvv.iloc[0]['title']=='':
                    pass;
#                     raise ValueError('Section Error')
                
                section_dict[k][kk][kkk] = {'name':vvv.iloc[0]['title'], 'content':[]}
                for kkkk,vvvv in vvv.groupby('group'):
                    image, caption = None, None
                    
                    image_df = vvvv[vvvv['image_url']!='']
                    if len(image_df)>0:
                        image, caption = image_df['image_url'].iloc[0], image_df['content'].iloc[0]
                    
                    section_dict[k][kk][kkk]['content'].append({
                        'image':image,
                        'caption':caption,
                        'group':vvvv['content'].tolist(),
                    })
                    
    return section_dict

def flatten_section_dict(section_dict):
    section_dict_flat = []

    for k,v in section_dict.items():
        for kk,vv in v.items():
            for kkk,vvv in vv.items():
                for i,x in enumerate(section_dict[k][kk][kkk]['content']):
                    section_dict_flat.append(((k,kk,kkk), i, ' '.join(x['group']).strip()))
    return section_dict_flat



class DocSearch:
    def __init__(self, chunk_df):
        self.df = chunk_df
        self.section_dict = make_section_dict(chunk_df)
        self.cliffs_dict = make_cliffs_dict(chunk_df)
        self.digest_dict = make_digest_dict(chunk_df)
        
        self.section_dict_flat = flatten_section_dict(self.section_dict)
        self.groups = [x[-1] for x in self.section_dict_flat]
        self.vecs = CountVectors(*self.groups)
        
    def query(self, qtext, topk=3, thresh=0.1):
        tt = torch.Tensor(self.query_groups(qtext))
        matches = [
            self.section_dict_flat[i]+(v,)
            for i,v in zip(tt.squeeze().topk(topk).indices, tt.squeeze().topk(topk).values)
            if v>thresh
        ]

        if len(matches)==0:
            print('No relevant sections found...')
        else:
            query_dict = {}
            
            for (k,kk,kkk), g, _, v in matches:
                if k not in query_dict.keys():
                    query_dict[k] = {}
                if kk not in query_dict[k].keys():
                    query_dict[k][kk] = {}
                if kkk not in query_dict[k][kk].keys():
                    query_dict[k][kk][kkk] = {
                        'name':self.section_dict[k][kk][kkk]['name'],
                        'content':[]
                    }
                    
                score = str(np.around(v.item(), 3))
                query_dict[k][kk][kkk]['content'].append({
                    'image':self.section_dict[k][kk][kkk]['content'][g]['image'],
                    'caption':self.section_dict[k][kk][kkk]['content'][g]['caption'],
                    'group':self.section_dict[k][kk][kkk]['content'][g]['group'],
                    'score':score
                })
            return query_dict
        
    def query_groups(self, qtext, rows=None):
        token_idxs = self.vecs.token_idxs(qtext)
        rows = np.array(rows) if rows is not None else slice(None)
        cols = np.array(token_idxs) if len(token_idxs)>0 else slice(None)
#         qvec = self.vecs.query(qtext)[:, cols]
#         kvec = self.vecs.vectors[rows,:][:, cols]
        qvec = normalize(self.vecs.query(qtext)[:, cols], norm='l1', axis=0)
        kvec = normalize(self.vecs.vectors[rows,:][:, cols], norm='l1', axis=0)
        sim = (kvec@qvec.T).toarray()
        return sim