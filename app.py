import streamlit as st
import re
import glob
import torch
import spacy 
# import en_core_web_sm
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead

import SessionState
from src.data.utils import reget, unique

device = 'cuda' if torch.cuda.is_available() else 'cpu'

re_padding = re.compile(r'(?<=^)[^\>\<]+(?=(\<|$))|(?<=\>)[^\>\<]+(?=(\<|$))')
re_lowers = re.compile(r'(?<=^)^[a-z]|(?<=[\.\?\!]\s)[a-z]')

import copy
from tqdm import tqdm

# @st.cache
def generate_summary(
    context, model, tokenizer, 
    max_input_length=512,
    max_output_length=300, 
    min_output_length=50, 
    length_penalty=2.0, 
    num_beams=4, 
    early_stopping=True
):
    model.to(device)
    inputs = tokenizer.encode(
        "summarize: " + context, 
        return_tensors="pt", 
        max_length=max_input_length, 
        truncation=True
    ).to(device)
    
    outputs = model.generate(
        inputs, 
        max_length=max_output_length, 
        min_length=max_output_length, 
        length_penalty=length_penalty, 
        num_beams=num_beams,
        early_stopping=True
    )

    summary = re_padding.search(tokenizer.decode(outputs[0])).group().strip()
    summary = re_lowers.sub(lambda x: x.group()[0].upper()+x.group()[1:], summary)
    return summary


    
def make_summaries(section_dict, model_t5, tokenizer_t5, save_dir):
    sub_summary = copy.deepcopy(section_dict)
    for k,v in tqdm(sub_summary.items()):
        for kk,vv in v.items():
            for kkk,vvv in vv.items():
                for i,x in enumerate(sub_summary[k][kk][kkk]['content']):

                    sub_summary[k][kk][kkk]['content'][i] = generate_summary(
                        ' '.join(x).strip(), model_t5, tokenizer_t5,
                        max_output_length=150
                    )
    
    summary = copy.deepcopy(sub_summary)
    for k,v in tqdm(summary.items()):
        for kk,vv in v.items():
            for kkk,vvv in vv.items():

                summary[k][kk][kkk]['content'] = generate_summary(
                    ' '.join(summary[k][kk][kkk]['content']).strip(), model_t5, tokenizer_t5,
                    max_output_length=350
                )
                
    with open(os.path.join(save_dir, 'sub_summary.pkl'), "wb") as f:
        pickle.dump(sub_summary, f)
        
    with open(os.path.join(save_dir, 'summary.pkl'), "wb") as f:
        pickle.dump(summary, f)

# @st.cache
def generate_questions(answers, context, model_qa, tokenizer_qa, max_length=64):
    
    model_qa.to(device)
    
    qa_pairs = []
    for answer in answers:
        input_text = "answer: %s  context: %s </s>" % (answer, context)

        features = tokenizer_qa([input_text], return_tensors='pt')
        output = model_qa.generate(input_ids=features['input_ids'].to(device), 
                                attention_mask=features['attention_mask'].to(device),
                                max_length=max_length)
        question = reget(r'(?<=\:\s).+(?=\<)', tokenizer_qa.decode(output[0]))
        qa_pairs.append((question, answer))

    return qa_pairs

# @st.cache
def generate_keywords(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    words = [str(x) for x in doc]
    labels = [str(x.ent_type_) for x in doc]
    ents, names = [], []
    for i, (word, label) in enumerate(zip(words, labels)):
        if label == '':
            continue
        elif labels[i]==labels[i-1] and len(ents)>0:
            ents[-1]+=' '+word
        else:
            ents.append(word)
            names.append(label)
    return ents, names

# @st.cache
def make_contexts(section_dict):
    contexts = copy.deepcopy(section_dict)
    for k,v in tqdm(contexts.items()):
        for kk,vv in v.items():
            for kkk,vvv in vv.items():
                contexts[k][kk][kkk]['content'] = []
                context = []
                n_words = 0
                for i,x in enumerate(section_dict[k][kk][kkk]['content']):
                    for j,chunk in enumerate(x):
                        if len(context)<5 and n_words<250:
                            context.append(chunk)
                            n_words += len(chunk.split())
                        else:
                            text = ' '.join(context).strip()
                            words, tags = generate_keywords(text)
                            contexts[k][kk][kkk]['content'].append({
                                'context':text,
                                'keywords':words,
                                'tags':tags
                            })
                            context = []
                            n_words = 0
    return contexts

# @st.cache
def make_questions(contexts, model_qa, tokenizer_qa, save_dir):
    questions = copy.deepcopy(contexts)
    for k,v in tqdm(questions.items()):
        for kk,vv in v.items():
            for kkk,vvv in vv.items():
                for i,x in enumerate(questions[k][kk][kkk]['content']):
                    qa_pairs = generate_questions(x['keywords'], x['context'], model_qa, tokenizer_qa)
                    x['questions'] = [
                        {'question':q, 'answer':'{} ({})'.format(a, tag)} 
                        for (q, a), tag in zip(qa_pairs, x['tags'])
                    ]
                    _ = x.pop('keywords')
                    _ = x.pop('tags')
    with open(os.path.join(save_dir, 'questions.pkl'), "wb") as f:
        pickle.dump(questions, f)
    
import os
import pickle
from src.data.pdf_processor import PDFProcessor
from src.data.search import make_section_dict, DocSearch
from src.data.dif_processor import DIFProcessor

from src.data.utils import box_edges
from src.data.pdf_featurizer import make_mask, make_spacing_back, make_spacing_front, make_chunks
from src.data.pdf_featurizer import make_sections, make_section_segments

def get_name_cat(cat):
    return {
        reget(r'[a-zA-Z]+(?=\_[A-Z])', x).upper():x 
        for x in glob.glob('data/dod/*_{}'.format(cat))
    }

def tm_to_section_dict(data_dir):
    tm = PDFProcessor(data_dir=data_dir)
    tm.load_pdf(fix_boxes=True)

    tm.add_feature(make_mask, 'mask')
    tm.add_feature(make_spacing_back, 'spacing')
    tm.add_feature(make_chunks, 'chunks', 'groups', 'types')
    tm.add_feature(lambda x: make_sections(x, verbose=False), 'sections', 'subsections', 'subsubsections')
    return make_section_dict(tm.df)
    
def run():
    model_t5 = AutoModelWithLMHead.from_pretrained("t5-base")
    tokenizer_t5 = AutoTokenizer.from_pretrained("t5-base")

    tokenizer_qa = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model_qa = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    
    dirs = {'TM':get_name_cat('TM'), 'DIF':get_name_cat('DIF')}
    state = SessionState.get(TM={}, DIF={}, display={}, summary={}, questions={})
    
    cat = 'TM'
    st.sidebar.subheader('Data Processor')

    process = st.sidebar.selectbox(
        "Unprocessed Data", 
        sorted([x for x in dirs[cat] if x not in state.__dict__[cat].keys()], reverse=True)
    )
    
    if st.sidebar.button("Process Data"):
        state.__dict__[cat][process]=tm_to_section_dict(data_dir=dirs[cat][process])
        SessionState.rerun()

    st.sidebar.subheader('Data Viewer')
    
    view = st.sidebar.selectbox("Processed Data", sorted(list(state.__dict__[cat].keys())))
        
    st.title('Data Explorer')
    
    if view is None:
        st.sidebar.write('First you must use the Data Processor to process and select a TM or DIF course')
    else:
#         view_type = st.sidebar.radio('View Style', ('by section', 'by search'), 0)
#         st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        doc_search = DocSearch(state.__dict__[cat][view])
        state.__dict__['summary'][view] = os.path.exists(os.path.join(dirs[cat][view], 'summary.pkl'))
        state.__dict__['questions'][view] = os.path.exists(os.path.join(dirs[cat][view], 'questions.pkl'))
    
        if st.sidebar.button("View Document"):
            state.display = doc_search.section_dict
            st.write(state.display)
            
            
        if not state.__dict__['summary'][view]:
            if st.sidebar.button("Make Summaries"):
                st.sidebar.write('this may take a while...')
                make_summaries(doc_search.section_dict, model_t5, tokenizer_t5, dirs[cat][view])
                SessionState.rerun()
                
        else:
            if st.sidebar.button("View Summaries"):
                with open(os.path.join(dirs[cat][view], 'summary.pkl'), "rb") as f:                    
                    state.display = pickle.load(f)
                    st.write(state.display)
            if st.sidebar.button("View Description"):
                with open(os.path.join(dirs[cat][view], 'sub_summary.pkl'), "rb") as f:                    
                    state.display = pickle.load(f)
                    st.write(state.display)
                    
        if not state.__dict__['questions'][view]:
            if st.sidebar.button("Make Questions"):
                st.sidebar.write('this may take a while...')
                make_questions(make_contexts(doc_search.section_dict), model_qa, tokenizer_qa, dirs[cat][view])
                SessionState.rerun()
                
        else:
            if st.sidebar.button("View Questions"):
                with open(os.path.join(dirs[cat][view], 'questions.pkl'), "rb") as f:                    
                    state.display = pickle.load(f)
                    st.write(state.display)

        topk = st.sidebar.slider("Number of Results", 0, 10, 3)
        thresh = st.sidebar.slider("Threshold", 0., 0.3, 0.05, 0.0025)

        query = st.sidebar.text_area("Search Field", "")
        
        if st.sidebar.button("Search"):
            state.display = doc_search.query(query, topk=topk, thresh=thresh)
            st.write(state.display)


if __name__ == "__main__":
    run()
