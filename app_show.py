import streamlit as st
import re
import glob
# import spacy 

import SessionState
from src.data.utils import reget, unique
    
import os
import pickle
from src.data.pdf_processor import PDFProcessor
from src.data.search import DocSearch
from src.data.dif_processor import DIFProcessor
from src.data.utils_io import load_pickle, save_pickle

def get_name_cat(cat):
    return {
        reget(r'[a-zA-Z]+(?=\_[A-Z])', x).upper():x 
        for x in glob.glob('data/dod/*_{}'.format(cat))
    }

def tm_to_chunks(data_dir):
    tm = PDFProcessor(data_dir=data_dir)
    tm.load_chunks()
    return tm.df

import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())



def P(word, WORDS): 
    "Probability of `word`."
    N=sum(WORDS.values())
    return WORDS[word] / N

def correction(word, WORDS): 
    "Most probable spelling correction for word."
    return max(candidates(word, WORDS), key=lambda x: P(x, WORDS))

def candidates(word, WORDS): 
    "Generate possible spelling corrections for word."
    return (known([word], WORDS) or known(edits1(word), WORDS) or known(edits2(word), WORDS) or [word])

def known(words, WORDS): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correct_query(text, WORDS):
    text = [t.strip() for t in text.split() if len(t.strip())>0]
    corrected = []
    fixed = False
    for t in text:
        if t in WORDS:
            corrected.append(t)
        else:
            corrected.append(correction(t, WORDS))
            fixed = True
    if fixed:
        return ' '.join(text), ' '.join(corrected)
    else:
        return ' '.join(text), None
    
def run():
    dirs = {'TM':get_name_cat('TM'), 'DIF':get_name_cat('DIF')}
    state = SessionState.get(TM={}, DIF={}, display={}, WORDS=Counter())
    
    cat = 'TM'
    st.sidebar.header('View Document')

    view = st.sidebar.selectbox(
        "Select Document", 
        sorted([name for name, path in dirs[cat].items() if os.path.exists(os.path.join(path, 'chunks.tsv'))], reverse=True)
    )
   

    view_type = st.sidebar.radio(
        'View Document', 
        ('Full Text', "Readers Digest", "Cliffs Notes", 'Checks on Learning'), 
        0
    )
    
    st.title(view_type)

    doc_search = DocSearch(tm_to_chunks(data_dir=dirs[cat][view]))
    state.WORDS = Counter(words(' '.join(doc_search.df['content'])))
    if view_type=='Full Text':
        state.display = doc_search.section_dict
    elif view_type=="Readers Digest":
        state.display = doc_search.digest_dict
    elif view_type=="Cliffs Notes":
        state.display = doc_search.cliffs_dict
    elif view_type=='Checks on Learning':
        if os.path.exists(os.path.join(dirs[cat][view], 'questions.pkl')):
            state.display = load_pickle(os.path.join(dirs[cat][view], 'questions.pkl'))
        else:
            state.display = {}
    
    st.sidebar.header('Search Document')
    
    topk = st.sidebar.slider("Number of Search Results", 0, 10, 3)
    thresh = st.sidebar.slider("Search Relevance Threshold", 0., 0.3, 0.05, 0.0025)

    query = st.sidebar.text_area("Search Field", "")

    if st.sidebar.button("Search"):
        text, corrected = correct_query(query, state.WORDS)
        if corrected is not None:
            st.subheader('\*\*\*SUSPECTED TYPO: {} -> {}\*\*\*'.format(text, corrected))
            state.display = doc_search.query(corrected, topk=topk, thresh=thresh)
        else:
            state.display = doc_search.query(text, topk=topk, thresh=thresh)
        
    st.sidebar.write("Note: Currently the search function can only search through full text")

    for k,v in state.display.items():
        for kk,vv in v.items():
            for kkk,vvv in vv.items():
                with st.beta_expander(str((k,kk,kkk))+' '+vvv['name']):
                    for i,x in enumerate(vvv['content']):
                        if isinstance(x, dict):
                            if 'score' in x:
                                st.write(x['score'])
                            elif 'image' in x and x['image'] is not None:
                                st.image(x['image'])
                                st.write(x['caption'])
                                
                            if len(x['group'])>0:
                                st.write(x['group'])
                        else:
                            st.write(x)

if __name__ == "__main__":
    run()
