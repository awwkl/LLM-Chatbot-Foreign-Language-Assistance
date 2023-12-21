import os
import json
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer
from parrot import Parrot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Variables to change ===
language = 'German'
# ===========================


def create_queries_for_bard(lang='English'):    
    # append_line = f'(in {lang}). Give me examples, pronunciations, and tips for someone new to the {lang} language.'
    append_line = f'(in {lang}, for someone new to the {lang} language). Give me examples and format each example like this: {lang} words: (English translation, English pronunciation). Use point format, and not table format. Afterwards, give me some tips specific to the question I asked, not generic tips for learning {lang}. '
    
    query_file_path = '0-data/train_queries.txt'
    query_file = open(query_file_path, 'r')
    for line in query_file.readlines():
        if not line.startswith('['):
            continue
        
        query_tag = line[:4].replace('[', '').replace(']', '')
        query = line.split('] ')[1].strip()
        query = query.split('||')[0].strip()
        query_to_bard = f'{query} {append_line}'
        
        print(query_tag)
        print(query_to_bard)

def paraphrase(text, model, tokenizer, max_length=128, num_return_sequences=10):
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(device)
    generated_ids = model.generate(input_ids=input_ids, num_return_sequences=num_return_sequences, num_beams=num_return_sequences, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=999.0, length_penalty=1.0, early_stopping=True)
    # generated_ids = model.generate(input_ids=input_ids, num_return_sequences=num_return_sequences, num_beams=5)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    return preds


def create_paraphrased_queries():
    # parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
    
    # hf_name = 'eugenesiow/bart-paraphrase'
    hf_name = 'mrm8488/t5-small-finetuned-quora-for-paraphrasing'
    model = AutoModelWithLMHead.from_pretrained(hf_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    paraphrased_queries_dict = {}
    
    query_file_path = '0-data/train_queries.txt'
    query_file = open(query_file_path, 'r')
    for line in query_file.readlines():
        if not line.startswith('['):
            continue
        
        query_tag = line[:4]
        query_full = line.split('] ')[1].strip()
        
        query_long  = query_full.split('||')[0].strip()
        query_short_1 = query_full.split('||')[1].strip()
        query_short_2 = query_full.split('||')[2].strip()
        
        # paraphrased_queries_list = paraphrase(query_long, model, tokenizer) + [query_short_1, query_short_2]
        paraphrased_queries_list = paraphrase(query_long, model, tokenizer)
        print('-'*100)
        print(f'{query_tag}:', query_long)
        print('-'*100)
        for paraphrased_query in paraphrased_queries_list:
            print(paraphrased_query)
        
        paraphrased_queries_dict[query_tag] = {
            'query': query_long,
            'paraphrased_queries_list': paraphrased_queries_list,
        }
        
        # paraphrased_queries_list = parrot.augment(input_phrase=query,
        #                                           do_diverse=True,
        #                                         #   max_return_phrases = 10, 
        #                                           use_gpu=True,
        # )
        # paraphrased_queries_list = [x for (x,num) in paraphrased_queries_list]
        
    json_out_path = '0-data/paraphrased_query_list.json'
    with open(json_out_path, 'w') as outfile:
        json.dump(paraphrased_queries_dict, outfile)
        
create_queries_for_bard(language)
# create_paraphrased_queries()