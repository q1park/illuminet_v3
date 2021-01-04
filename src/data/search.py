import numpy as np
import torch

from sklearn.preprocessing import normalize
from src.nlp.tfidf import CountVectors, TFIDFVectors
from src.data.pdf_featurizer import make_section_segments

def make_section_dict(line_df):
    section_dict = {}
    section_segments = make_section_segments(line_df, verbose=False)

    clean_df = line_df.drop(columns=['c0', 'size', 'face', 'marker', 'line', 'page', 'x0', 'y0', 'x1', 'y1', 'spacing'])
    clean_df = clean_df[(clean_df['mask']==0)&(clean_df['sections']>=0)&(clean_df['subsections']>=0)&(clean_df['types']>0)]

    for k,v in clean_df.groupby('sections'):
        section_dict[k] = {}

        for kk,vv in v.groupby('subsections'):
            section_dict[k][kk] = {}

            for kkk,vvv in vv.groupby('subsubsections'):
                section_dict[k][kk][kkk] = {'name':section_segments[(k,kk,kkk)][0], 'content':[]}

                for kkkk,vvvv in vvv.groupby('groups'):
                    content = [' '.join(v['content'].tolist()).strip() for k,v in vvvv.groupby('chunks')]
                    section_dict[k][kk][kkk]['content'].append(content)
    return section_dict

def flatten_section_dict(section_dict):
    section_dict_flat = []

    for k,v in section_dict.items():
        for kk,vv in v.items():
            for kkk,vvv in vv.items():
                for i,x in enumerate(section_dict[k][kk][kkk]['content']):
                    section_dict_flat.append(((k,kk,kkk), i, ' '.join(x).strip()))
    return section_dict_flat

class DocSearch:
    def __init__(self, section_dict):
        self.section_dict = section_dict
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
                query_dict[k][kk][kkk]['content'].append({score:self.section_dict[k][kk][kkk]['content'][g]})
            return query_dict
        
    def query_groups(self, qtext, rows=None):
        token_idxs = self.vecs.token_idxs(qtext)
        rows = np.array(rows) if rows is not None else slice(None)
        cols = np.array(token_idxs) if len(token_idxs)>0 else slice(None)

        qvec = normalize(self.vecs.query(qtext)[:, cols], norm='l1', axis=0)
        kvec = normalize(self.vecs.vectors[rows,:][:, cols], norm='l1', axis=0)
        sim = (kvec@qvec.T).toarray()
        return sim