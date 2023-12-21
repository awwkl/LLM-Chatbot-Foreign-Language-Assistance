import os
import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from utils import get_doc_embeddings, get_most_similar_doc_tag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Variables that can be varied  ===
chosen_doc_word_len = 30
hf_name = 'awwkl/multi-qa-MiniLM-L6-cos-v1-cs425' # this is the trained model hosted on HuggingFace
target_language = 'German'

# inference_queries = ['basic phrases', 'ask for directions', 'get help', 'order food']
inference_queries = ['what are some basic phrases', 'how to ask for directions', 'how to ask for help', 'how to order food']
# =====================================

# === Set up global variables once, because they are costly to repeat ===
model = SentenceTransformer(hf_name).to(device)
doc_embeddings = get_doc_embeddings(model, chosen_doc_word_len=chosen_doc_word_len)
# =====================================


def get_doc_string(language, doc_tag):
    doc_path = os.path.join('0-data/lang_documents', language, f'{doc_tag}.txt')
    f = open(doc_path, 'r')
    doc_string = f.read()
    return doc_string

    
def retrieve_document(inference_query, language):
    most_similar_doc_tag = get_most_similar_doc_tag(model, inference_query, doc_embeddings)
    doc_string = get_doc_string(language, most_similar_doc_tag)
    
    return doc_string

    print(f'--- inference_query: {inference_query} ---')
    print('most_similar_doc_tag:', most_similar_doc_tag)
    print('doc_string:', doc_string[:100])
    

for inference_query in inference_queries:
    retrieve_document(inference_query, target_language)