import re
import glob
import torch
import spacy 
import en_core_web_sm
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead
from src.data.utils import reget

device = 'cuda' if torch.cuda.is_available() else 'cpu'

re_padding = re.compile(r'(?<=^)[^\>\<]+(?=(\<|$))|(?<=\>)[^\>\<]+(?=(\<|$))')
re_lowers = re.compile(r'(?<=^)^[a-z]|(?<=[\.\?\!]\s)[a-z]')

# @st.cache
def generate_summary(
    context, model, tokenizer, 
    max_input_length=512,
    max_output_length=300, 
    min_output_length=25, 
    length_penalty=2.0, 
    repetition_penalty=2.0,
    temperature=1.0,
    num_beams=1, 
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
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        temperature=temperature,
        early_stopping=True
    )

    summary = re_padding.search(tokenizer.decode(outputs[0])).group().strip()
    summary = re_lowers.sub(lambda x: x.group()[0].upper()+x.group()[1:], summary)
    return summary

def make_chunk_mask(chunk_df):
    context_mask = []
    for i,row in chunk_df.iterrows():
        if row['title']=='' and (len(row['content'].split())<3 or re.search(r'\s[a-z]{2}', row['content']) is None):
            context_mask.append(1)
        else:
            context_mask.append(0)
    chunk_df['context_mask'] = context_mask
    return chunk_df

def make_chunk_summaries(chunk_df, model_t5, tokenizer_t5):
    chunk_summaries = []
    for i,row in tqdm(chunk_df.iterrows(), total=len(chunk_df)):
        if len(row['content'].split())>50:
            chunk_summaries.append(generate_summary(row['content'], model_t5, tokenizer_t5, max_output_length=50, repetition_penalty=2.0))
        else:
            chunk_summaries.append(row['content'])
    chunk_df['chunk_summary'] = chunk_summaries
    return chunk_df

def make_group_summaries(chunk_df, model_t5, tokenizer_t5):
    group_summaries = []
    
    df_groupby = chunk_df.groupby('group')
    for k,v in tqdm(df_groupby, total=len(df_groupby)):
        len_group = len(v)
        context = ' '.join(v[(v['title']=='')&(v['context_mask']==0)]['chunk_summary']).strip()
        
        if len(context.split())>100:
            summary = generate_summary(context, model_t5, tokenizer_t5, max_output_length=100, repetition_penalty=2.0)
        else:
            summary = context
        group_summaries.extend([summary]+['']*(len_group-1))
    chunk_df['group_summary'] = group_summaries
    return chunk_df

def make_section_summaries(chunk_df, model_t5, tokenizer_t5):
    section_summaries = []
    
    df_sectionby = chunk_df.groupby('section_tag')
    for k,v in tqdm(df_sectionby, total=len(df_sectionby)):
        len_section = len(v)
        context = ' '.join(v[v['group_summary']!='']['group_summary']).strip()
        
        if len(context.split())>150:
            summary = generate_summary(context, model_t5, tokenizer_t5, max_output_length=300, repetition_penalty=2.0)
        else:
            summary = context
        section_summaries.extend([summary]+['']*(len_section-1))
    chunk_df['section_summary'] = section_summaries
    return chunk_df

re_stop = re.compile(r'^the\s|^this\s|^a\s', re.IGNORECASE)

def filter_noun_phrases(noun_phrases):
    wordlist = []
    stemset = set()

    for x in noun_phrases:
        stemmed = re_stop.sub('', x).strip().lower()
        if stemmed not in stemset:
            wordlist.append(x)
            stemset.add(stemmed)
    return wordlist

def generate_noun_phrases(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    noun_phrases = filter_noun_phrases([chunk.text for chunk in doc.noun_chunks])
    return noun_phrases

def generate_questions(text, model_qa, tokenizer_qa, max_length=64):
    model_qa.to(device)
    noun_phrases = generate_noun_phrases(text)
    
    if len(noun_phrases)<=7:
        return []
    
    inputs = [tokenizer_qa([x], return_tensors='pt') for x in noun_phrases]

    embeddings = torch.cat([
        model_qa.encoder(**{
            k:v.to(device) for k,v in t.items()
        }).last_hidden_state.mean(dim=1)
        for t in inputs], 
        dim=0
    )
    
    closest = [
        x.topk(3).indices.cpu().detach().tolist() 
        for x in (embeddings@embeddings.T).sqrt().fill_diagonal_(0.)
    ]
    
    answers = {np:[noun_phrases[i] for i in close] for np, close in zip(noun_phrases, closest)}
    qa_pairs = []
    
    for answer, wrongs in answers.items():
        input_text = "answer: %s  context: %s </s>" % (answer, text)

        features = tokenizer_qa([input_text], return_tensors='pt')
        output = model_qa.generate(
            input_ids=features['input_ids'].to(device), 
            attention_mask=features['attention_mask'].to(device),
            max_length=max_length
        )
        question = reget(r'(?<=\:\s).+(?=\<)', tokenizer_qa.decode(output[0]))
        qa_pairs.append({'question':question, 'answers':[answer]+wrongs})
    return qa_pairs

def generate_group_questions(group_df):
    title = group_df[group_df['title']!='']['title'].iloc[0]
    group_df = group_df[(group_df['context_mask']==0)&(group_df['title']=='')]
    contexts = []
    context = []
    last_marker = -1

    for i,row in group_df.iterrows():
        chunk = row['content'].split()

        if len(context)>50:
            contexts.append(' '.join(context))
            context = chunk
        elif len(context)<10 and row['marker']==last_marker:
            context.extend(chunk)
        elif len(chunk)<30 and row['marker']==last_marker:
            context.extend(chunk)
        else:
            contexts.append(' '.join(context))
            context = chunk
        last_marker = row['marker']

    contexts.append(' '.join(context))
    questions = [{'context':x, 'questions':generate_questions(x, model_qa, tokenizer_qa)} for x in contexts]
    return questions

def make_question_dict(chunk_df):   
    question_dict = {}

    for k,v in tqdm(chunk_df.groupby('section')):
        question_dict[k] = {}

        for kk,vv in v.groupby('subsection'):
            question_dict[k][kk] = {}

            for kkk,vvv in vv.groupby('subsubsection'):
#                 if vvv.iloc[0]['title']=='':
#                     raise ValueError('Section Error')
                
                question_dict[k][kk][kkk] = {
                    'name':vvv.iloc[0]['title'], 
                    'content':generate_group_questions(vvv)}
    return question_dict