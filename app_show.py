import streamlit as st
import re
import glob
import spacy 
import en_core_web_sm

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
    st.sidebar.subheader('Data Processor')

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
    
    st.title('Data Explorer')

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
    
    topk = st.sidebar.slider("Number of Results", 0, 10, 3)
    thresh = st.sidebar.slider("Threshold", 0., 0.3, 0.05, 0.0025)

    query = st.sidebar.text_area("Search Field", "")

    if st.sidebar.button("Search"):
        state.display = doc_search.query(query, topk=topk, thresh=thresh)
        
    st.write(state.display)

if __name__ == "__main__":
    run()