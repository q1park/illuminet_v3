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
    
def run():
    dirs = {'TM':get_name_cat('TM'), 'DIF':get_name_cat('DIF')}
    state = SessionState.get(TM={}, DIF={}, display={})
    
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
        
    doc_search = DocSearch(tm_to_chunks(data_dir=dirs[cat][view]))
    
    st.title('Document Viewer')

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
        state.display = doc_search.query(query, topk=topk, thresh=thresh)
        
    st.sidebar.write("Note: Currently the search function can only search through full text")
    st.write(state.display)

    for k,v in state.display.items():
        for kk,vv in v.items():
            for kkk,vvv in vv.items():
                with st.beta_expander(str((k,kk,kkk))+' '+vvv['name']):
                    st.write(vvv['content'])

if __name__ == "__main__":
    run()
