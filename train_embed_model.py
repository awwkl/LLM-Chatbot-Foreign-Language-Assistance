import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
train_batch_size = 4
n_epochs = 25
output_path = f'0-saved-models/multi-qa'

def get_train_examples():
    train_examples = []
    
    paraphrased_queries_path = '0-data/t5_paraphrased_query_list.json'
    with open(paraphrased_queries_path, 'r') as file:
        paraphrased_queries_dict = json.load(file)
        
        for query_label_ind, query_tag in enumerate(paraphrased_queries_dict):
            paraphrased_queries_list = paraphrased_queries_dict[query_tag]['paraphrased_queries_list']
            query_tag = query_tag.replace('[', '').replace(']', '')
            
            print('===', query_tag, '===')
            paraphrased_queries_list = paraphrased_queries_list[:-2]
            print(paraphrased_queries_list)
            
            for paraphrased_query in paraphrased_queries_list:
                input_example = InputExample(texts=[paraphrased_query], label=query_label_ind)
                train_examples.append(input_example)
    
    return train_examples

model = SentenceTransformer(hf_name).to(device)

train_examples = get_train_examples()
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.BatchAllTripletLoss(model=model)

model.fit([(train_dataloader, train_loss)], 
            epochs = n_epochs,
            output_path=output_path,
            checkpoint_path=output_path,
            checkpoint_save_steps = 200,
            # show_progress_bar=True,
)
