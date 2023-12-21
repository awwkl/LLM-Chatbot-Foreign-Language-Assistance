import os
import numpy as np
from sentence_transformers import util

chosen_doc_word_len = 99999

def get_doc_embeddings(model, language='English', chosen_doc_word_len=99999):
    doc_embeddings = {}
    
    query_tag_list = []
    query_tag_list += [f'b{num+1}' for num in range(7)]
    query_tag_list += [f'd{num+1}' for num in range(2)]
    query_tag_list += [f'h{num+1}' for num in range(3)]
    query_tag_list += [f'o{num+1}' for num in range(8)]

    embedding_list = []
    for query_tag in query_tag_list:
        fpath = os.path.join('0-data', 'lang_documents', language, f'{query_tag}.txt')
        f = open(fpath, 'r')

        doc_string = f.read()
        print('doc len:', len(' '.join(doc_string.split())))
        doc_string = ' '.join(doc_string.split()[:chosen_doc_word_len]) # truncate words
        doc_embed = model.encode(doc_string)
        embedding_list.append(doc_embed)
    
    doc_embeddings['query_tag_list'] = query_tag_list
    doc_embeddings['embedding_list'] = embedding_list
    return doc_embeddings

def get_most_similar_doc_tag(model, query, doc_embeddings):
    query_embed = model.encode(query)
    score_fn = util.dot_score
    # score_fn = util.cos_sim
    score_list = score_fn(query_embed, doc_embeddings['embedding_list'])[0].cpu().tolist()
    
    most_similar_ind = np.argmax(score_list)
    most_similar_doc_id = doc_embeddings['query_tag_list'][most_similar_ind]
    return most_similar_doc_id