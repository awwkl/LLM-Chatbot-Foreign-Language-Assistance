import os
import torch
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
from utils import get_doc_embeddings, get_most_similar_doc_tag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chosen_doc_word_len = 99999

# hf_name = 'sentence-transformers/bert-base-nli-mean-tokens'
# hf_name = 'sentence-transformers/msmarco-distilbert-dot-v5'
# hf_name = 'sentence-transformers/sentence-t5-base'
# hf_name = 'sentence-transformers/sentence-t5-large'

# hf_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
# hf_name = 'sentence-transformers/all-MiniLM-L6-v2'
# hf_name = 'sentence-transformers/all-mpnet-base-v2'
# hf_name = '0-saved-models/multi-qa/1000'
hf_name = 'awwkl/multi-qa-MiniLM-L6-cos-v1-cs425'

# === Set up global variables once, because they are costly to repeat ===
model = SentenceTransformer(hf_name).to(device)
doc_embeddings = get_doc_embeddings(model, chosen_doc_word_len=chosen_doc_word_len)
# =====================================



def eval_query_retrieval():
    y_true_list = []
    y_pred_list = []
    
    paraphrased_queries_path = '0-data/t5_paraphrased_query_list.json'
    with open(paraphrased_queries_path, 'r') as file:
        paraphrased_queries_dict = json.load(file)
        
        for query_label_ind, query_tag in enumerate(paraphrased_queries_dict):
            paraphrased_queries_list = paraphrased_queries_dict[query_tag]['paraphrased_queries_list']
            query_tag = query_tag.replace('[', '').replace(']', '')
            
            paraphrased_queries_list = paraphrased_queries_list[-2:]
            
            for paraphrased_query in paraphrased_queries_list:
                most_similar_doc_tag = get_most_similar_doc_tag(model, paraphrased_query, doc_embeddings)
                
                if not (query_tag == most_similar_doc_tag):
                    print(f'Error: {query_tag} ({paraphrased_query}) retrieved {most_similar_doc_tag}')

                y_true_list.append(query_tag)
                y_pred_list.append(most_similar_doc_tag)
    
    print(classification_report(y_true_list, y_pred_list, digits=3))
    print(accuracy_score(y_true_list, y_pred_list))
                
eval_query_retrieval()