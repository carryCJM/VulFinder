import os
import argparse
import shutil
import joblib
import tables
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_
from torch.nn.utils.rnn import pad_sequence
# import transformers
from transformers import LlamaForCausalLM, CodeLlamaTokenizer,AutoModelForCausalLM, LlamaModel
from transformers import (AutoTokenizer, T5Config, T5ForConditionalGeneration, RobertaTokenizer)


def get_template_by_dataset(dataset_name):
    ## a prompt can be described as template_l+text_data+template_r 
    template_l = "Task: Detect and locate any vulnerabilities in the following code function. \n" 
    template_r = " Question: Based on the code above, this function is \"___\", and the vulnerable line is \"___\"."
        
    return template_l,template_r



class RawTextData(data_.Dataset):
    def __init__(self, text,node_id):
        self.text = text
        self.node_id = node_id 
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.node_id[idx]

def pretrain_collate_fn(data_tuple):
    seq = [torch.tensor(sq[0]) for sq in data_tuple]
    node_id = [sq[1] for sq in data_tuple]
    seq = pad_sequence(seq, batch_first=True, padding_value=tokenizer.pad_token_id)
    node_id = torch.tensor(node_id)
    return seq, node_id

    
def build_pretrain_data(model, tokenizer, x_text, save_path, template_l,template_r, device, args, data_type):

    func_embedding_path = os.path.join(save_path, f'{data_type}func_embedding.h5')
    with tables.open_file(func_embedding_path, mode='w') as f:
        atom = tables.Float16Atom()
        f.create_earray(f.root, 'data', atom, (0, 4096))

    code_line_embedding_path = os.path.join(save_path, f"{data_type}code_line_embeddings.h5")

    with tables.open_file(code_line_embedding_path, mode='w') as f:
        atom = tables.Float16Atom()
        f.create_earray(f.root, 'data', atom, (0, 4096))

    feature_ls = list(x_text)
    feature_ls_ids = []
    line_counts = [] 
    original_line_counts = []

    template_l_id = tokenizer(template_l)['input_ids']
    template_r_id = tokenizer(template_r)['input_ids']

    for f in tqdm.tqdm(feature_ls):
        original_line_count = f.count('\n')  
        original_line_counts.append(original_line_count)

        code_encoded = tokenizer(f, padding=False, truncation=True, max_length=2048, add_special_tokens=True)['input_ids']

        if code_encoded[0] == 1:  
            code_encoded = code_encoded[1:]
        if code_encoded and code_encoded[-1] == 2:  

            code_encoded = code_encoded[:-1] 

        line_for_text = [i for i, id in enumerate(code_encoded) if id == 13]
        line_count = len(line_for_text)  
        line_counts.append(line_count)
        prompt_l_encoded = template_l_id
        if prompt_l_encoded and prompt_l_encoded[-1] == 2:  
            prompt_l_encoded = prompt_l_encoded[:-1] 

        prompt_r_encoded = template_r_id

        if prompt_r_encoded[0] == 1: 
            prompt_r_encoded = prompt_r_encoded[1:]
        
        full_input = prompt_l_encoded + code_encoded + prompt_r_encoded

        feature_ls_ids.append(full_input)

    nodedata_ = RawTextData(feature_ls_ids, line_counts)
    node_data_loader = DataLoader(nodedata_, batch_size=args.batch_size, shuffle=False, collate_fn=pretrain_collate_fn)
    lists = []
    for _ in range(1): 
        
        for (text, line_counts) in tqdm.tqdm(node_data_loader):
            with torch.no_grad():

                text = text[:, :].to(device)

                attention_mask = (text != tokenizer.pad_token_id).long().half()

                embeddings = model(input_ids=text, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state

                mean_last_hidden_state = torch.mean(embeddings, dim=1).detach().to(torch.float).cpu().numpy()

                line_break_positions = []
                for i, text_id in enumerate(text):
                    line_for_text = [j for j, id in enumerate(text_id) if id == 13]

                    line_break_positions.append(line_for_text)
                    lists.append((len(line_for_text) - 1))
                batch_line_tokens_embeddings = [] 
                for i in range(len(line_break_positions)):  
                    line_breaks = line_break_positions[i]
                    embeddings_for_sample = embeddings[i] 
                    sample_line_tokens_embeddings = []  

                    if len(line_breaks) > 1:
                        for j in range(len(line_breaks) - 1):
                            line_start = line_breaks[j] + 1  
                            line_end = line_breaks[j + 1]+1  
                            line_token_embeddings = embeddings_for_sample[line_start:line_end] 
                            if torch.isnan(torch.tensor(embeddings_for_sample)).any():
                                print("Input data contains NaN")

                            line_tokens_embedding = torch.mean(line_token_embeddings, dim=0).unsqueeze(0)
                            sample_line_tokens_embeddings.append(line_tokens_embedding)
                            if torch.isnan(torch.tensor(line_tokens_embedding)).any():
                                print("Input data contains NaN")
                    else:
                        print("no \n")
                        line_tokens_embedding = torch.mean(embeddings_for_sample, dim=0).unsqueeze(0)
                        sample_line_tokens_embeddings.append(line_tokens_embedding)

                    batch_line_tokens_embeddings.append(torch.cat(sample_line_tokens_embeddings, dim=0))
                

                line_break_embeddings = torch.cat(batch_line_tokens_embeddings, dim=0).unsqueeze(0).squeeze(0).detach().to(torch.float).cpu().numpy()

                with tables.open_file(func_embedding_path, mode='a') as f:
                    f.root.data.append(mean_last_hidden_state)

                with tables.open_file(code_line_embedding_path, mode='a') as f:
                    f.root.data.append(line_break_embeddings)

                torch.cuda.empty_cache() 
                
    np.save(os.path.join(save_path, f"{data_type}line_break_positions.npy"), lists)

    return code_line_embedding_path

def convert_tables_to_npy(save_path, data_type):
    func_embeddings_path = os.path.join(save_path, f"{data_type}func_embedding.h5")

    with tables.open_file(func_embeddings_path, mode='r') as f:
        func_embeddings = f.root.data.read()
    np.save(os.path.join(save_path, f"{data_type}func_embeddings.npy"), func_embeddings)


    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess text-attributed graph by LLMs to gain the token embedding')
    parser.add_argument('--dataset_name', type=str, default='bigvul_as', choices=['arxiv', 'instagram', 'reddit', 'bigvul','bigvul_mini', 'sliced', 'diversevul', 'sliced_mini','bigvul_all'])
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--plm_path', type=str, default='/root/autodl-model/CodeLlama-7b-hf', choices=['F:/a_bug_location/transformer-model/Salesforce/codet5p-220m/', '/root/autodl-fs/Salesforce/codet5p-220m/', 'D:/transformer_model/Salesforce/codet5-base/'], help='Path of the pre-trained model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use')
    parser.add_argument('--pretrain_save_path', type=str, default='/root/autodl-tmp/token_embedding/', help='Path to save pre-training data')
    parser.add_argument('--prompt_save_path', type=str, default='/root/autodl-tmp/prompt_embedding/', help='Path to save prompt embeddings')

    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    save_path = os.path.join(args.pretrain_save_path, args.dataset_name)

    model = LlamaModel.from_pretrained(args.plm_path, torch_dtype=torch.half ).to(torch.bfloat16).to(args.device)

    tokenizer = CodeLlamaTokenizer.from_pretrained(args.plm_path)
    tokenizer.pad_token = '[PAD]'
    model.resize_token_embeddings(len(tokenizer))
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    train_x_text = np.load(f'/root/autodl-tmp/{args.dataset_name}/train_x_test_all_noref.npy', allow_pickle=True )
    test_x_text = np.load(f'/root/autodl-tmp/{args.dataset_name}/test_x_test_all_noref.npy', allow_pickle=True )
    
    template_l, template_r = get_template_by_dataset(args.dataset_name)
    
    build_pretrain_data(model, tokenizer, train_x_text, save_path, template_l,template_r, args.device, args, "train")
    build_pretrain_data(model, tokenizer, test_x_text, save_path, template_l,template_r, args.device, args, "test")

    
    convert_tables_to_npy(save_path, 'train')
    convert_tables_to_npy(save_path, 'test')
