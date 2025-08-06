import torch
import numpy as np
import torch_geometric as tg
from torch_geometric.data import InMemoryDataset, download_url, Data
from sklearn.metrics import roc_auc_score,accuracy_score
from transformers import get_scheduler
from transformers import get_linear_schedule_with_warmup
import joblib
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import random
# from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from main.Utils import Utilities 
import torch.utils.data as data_
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from transformers import ( RobertaTokenizer, T5Config, T5ForConditionalGeneration)
from transformers import get_cosine_schedule_with_warmup
import pdb

from torch.autograd import Variable

from model import GraphAdapter, NoGraphAdapter,HierarchicalGradNormLoss,AdaptiveLossBalancing

import json
import networkx as nx
import tables

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pretrain_graph_bigvul_all(dataset_name,data_type):
    data_file_path = f'/root/autodl-tmp/{dataset_name}/{data_type}_all.jsonl'
    line_break_positions = np.load(f'/root/autodl-tmp/token_embedding/{dataset_name}/{data_type}line_break_positions.npy', 
                                 allow_pickle=True)
    code_line_embedding_path =  f'/root/autodl-tmp/token_embedding/{dataset_name}/{data_type}code_line_embeddings.h5'

    with tables.open_file(code_line_embedding_path, mode='r') as f:
        x = f.root.data.read()
    print("adada")
    print("code_line_embeddings:",x.shape)

    func_embedding_path = np.load(f'/root/autodl-tmp/token_embedding/{dataset_name}/{data_type}func_embeddings.npy')

    
    data_list = []
    j = 0

    with open(data_file_path, 'r') as file:
        json_lines = file.readlines()

    target_count = [0, 0]
    label_count = [0, 0]

    for i, line in enumerate(json_lines):
        data = json.loads(line)

        actual_node_nums = line_break_positions[i]
        
        y_full = data['label_per_line']
        y_ = torch.tensor(y_full[:actual_node_nums], dtype=torch.long)

        x_ = torch.tensor(x[j:j + actual_node_nums], dtype=torch.float32)
        j += actual_node_nums
        edges = data['edges_without_blank_lines']
        edge_type = data['edge_type']

        valid_edge_mask = []
        for src, dst in edges:
            if src < actual_node_nums and dst < actual_node_nums:
                valid_edge_mask.append(True)
            else:
                valid_edge_mask.append(False)
        
        valid_edges = [edge for edge, mask in zip(edges, valid_edge_mask) if mask]
        valid_edge_types = [et for et, mask in zip(edge_type, valid_edge_mask) if mask]
        
        if valid_edges:
            edge_index = torch.tensor(valid_edges).t().contiguous().long()
            edge_attr = torch.tensor(valid_edge_types, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.float32)
        
        target = data['target']

        func_ = torch.tensor(func_embedding_path[i], dtype=torch.float32)

        if data_type == 'test':
            index = data['index']
            data_obj = Data(x=x_, y=y_, index = index,func=func_, edge_index=edge_index, edge_attr=edge_attr, target=target)
        else:
            data_obj = Data(x=x_, y=y_, func=func_, edge_index=edge_index, edge_attr=edge_attr, target=target)


        target_count[target] += 1
        label_count[0] += len(y_) - y_.sum()
        label_count[1] += y_.sum()
        data_list.append(data_obj)

    print("data_list length:",len(data_list))
    print("target_count:",target_count)
    print("label_count:",label_count)

    return data_list


# 预训练模块
def train_graph_adapter(args):
    dataset_name = args.dataset_name 
    llm_shape = args.llm_shape
    hiddensize_gnn = args.hiddensize_gnn
    hiddensize_fusion = args.hiddensize_fusion 
    normal_hidden_size = args.normal_hidden_size 
    dropout = args.dropout

    num_layers = args.num_layers 
    batch_size = args.batch_size 
    batch_size_eval = args.batch_size_eval
    batch_size_test = args.batch_size_test 
    pos_weight_set = args.pos_weight_set
    learning_ratio= args.learning_ratio 
    weight_decay = args.weight_decay
    max_epoch = args.max_epoch
    warm_up_ratio = args.warm_up_ratio
    use_GNN = args.use_GNN
    loss_type = args.loss_type 
    alpha = 0.5
    beta = 0.5

    set_random_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    GNN_type = args.GNN_type

    num_training_steps = args.max_epoch
    
    global train_code_line_embeddings
    global train_code_labels
    global val_code_line_embeddings
    global val_code_line_ids
    global val_code_labels
    global val_code_embeddings
    global test_code_line_embeddings
    global test_code_line_ids
    global test_code_labels
    global test_code_embeddings
    
    save_path = f'/root/autodl-fs/save_models/'
 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    joblib.dump(args,f'{save_path}model_args.pkl')
    
    logger = logging.getLogger()
    
    file_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=file_fmt, filename=f"{save_path}log.txt", filemode="a")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)
    console_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    fmt1 = logging.Formatter(fmt=console_fmt)
    console_handler.setFormatter(fmt=fmt1)
    logger.addHandler(console_handler)
    
    logging.info(f'save_path:{save_path}')
    
    logging.info(f"load load llm pretrain data, dataset_name:{dataset_name}")

    if args.do_train:
        train_data= load_pretrain_graph_bigvul_all(dataset_name,'train')
        val_data = load_pretrain_graph_bigvul_all(dataset_name,'val')
    if args.do_test:
        test_data = load_pretrain_graph_bigvul_all(dataset_name,'test')
        
    logging.info('load_data...OK')

    

    
    if args.do_train:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, follow_batch=['x'])
        val_loader = DataLoader(val_data, batch_size=batch_size_eval, shuffle=False, follow_batch=['x'])
    if args.do_test:
        test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False, follow_batch=['x'])
    
    logging.info('data_loader...OK')
    
    if args.do_train:
        total_steps =  len(train_loader) * max_epoch 
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_ratio, weight_decay=weight_decay)
        # 换一个带衰减的学习率
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps,
                                                       num_training_steps = total_steps)

    elif loss_type == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()

    criterion_label = torch.nn.CrossEntropyLoss()

    
    model = model.to(device)
    best_jaccard_index = 0
    if args.do_train:
        prev_losses = None
        for epoch in range(max_epoch):
            curr_losses = [0.0, 0.0] 
            total_loss = []
            count = 0
            model.train()
            for data_ in tqdm.tqdm(train_loader,ncols=50):
                data_ = data_.to(device)

                optimizer.zero_grad() 
                target_logits, line_logits = model(data_, training = True)
                
                loss_function = loss_fn(target_logits, data_.target.to(device))

                loss_line = criterion_label(line_logits, data_.y.to(device))
                
                loss = alpha * loss_function + beta * loss_line

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += [loss.item()*batch_size]


                lr_scheduler.step()

            logger.info("\n***** Running evaluation *****")
            total_eval_loss = []
            
            generated_texts = []
            groundtruth_texts = []
            generated_line_texts = []
            groundtruth_line_texts = []

            line_logits_per = []
            line_labels_per = []
            line_logits_per_all = []
            line_labels_per_all = []
            with torch.no_grad():
                model.eval()
                for data_ in tqdm.tqdm(val_loader,ncols=50):
                    data_ = data_.to(device)
                    target_logits, line_logits = model(data_,training = False)
                   
                    loss_function = loss_fn(target_logits, data_.target.to(device))

                    loss_line = criterion_label(line_logits, data_.y.to(device))
                    
                    loss = alpha * loss_function + beta * loss_line
                    
                    total_eval_loss += [loss.item()*batch_size]
                   
                    target_out = torch.argmax(target_logits,dim=1).float() 
                    generated_texts.append(target_out)
                    groundtruth_texts.append(data_.target)

                    

                    line_pre = torch.sigmoid(line_logits)

                    line_pre_out = torch.argmax(line_pre, dim=1).float()

                    generated_line_texts.append(line_pre_out)
                    groundtruth_line_texts.append(data_.y)

                    filter_correct_function_predictions_logit_label(line_logits_per,line_labels_per,target_out, data_.target,line_logits,data_.y,data_.batch, all = False, saved = False) 
                    filter_correct_function_predictions_logit_label(line_logits_per_all,line_labels_per_all,target_out, data_.target,line_logits,data_.y,data_.batch, all = True, saved = False) 


            result_line = evaluate_code_line(line_logits_per,line_labels_per, line_logits_per_all,line_labels_per_all)

            val_line = evaluate_code_only_coarse(generated_line_texts,groundtruth_line_texts,dataset_name,dev_type='test')

            val = evaluate_code_only_coarse(generated_texts,groundtruth_texts,dataset_name, dev_type='val')

            logging.info(f'epoch: {epoch} , loss: {np.sum(total_loss)}, eval loss: {np.sum(total_eval_loss)}')
            logging.info("function level result")
            for key in sorted(val.keys()):
                logging.info("  %s = %s", key, str(round(val[key], 6)))
            logging.info("line level classification result")
            for key in sorted(val_line.keys()):
                logging.info("  %s = %s", key, str(round(val_line[key], 6)))
            logging.info("line level rank result")
            for key in sorted(result_line.keys()):
                value = result_line[key]
                if isinstance(value, list):
                    logging.info("  %s = %s", key, str(value))
                else:
                    logging.info("  %s = %s", key, str(round(result_line[key], 6)))

            if epoch == 149 or epoch==199:
                torch.save(model.state_dict(),save_path+f'save_model_{epoch}.pkl')
            
            if val["f1_score"]>= best_jaccard_index  and epoch>=20 and epoch !=149  and epoch != 199:
                best_jaccard_index = val["f1_score"]
                torch.save(model.state_dict(),save_path+f'best_model.pkl')


    if args.do_test:

        logger.info("\n***** test *****")    
        
        epoch = -1
        path = save_path+f'best_model.pkl'
        logger.info(f"load best_model")
        model.load_state_dict(torch.load(path)) 

        generated_texts = []
        groundtruth_texts = []

        generated_line_texts = []
        groundtruth_line_texts = []

        line_logits_per = []
        line_labels_per = []
        line_logits_per_all = []
        line_labels_per_all = []


        with torch.no_grad():
            model.eval()
            total_test_loss = []
            for data_ in tqdm.tqdm(test_loader,ncols=50):
                data_ = data_.to(device)
                line_logits = model(data_, training = False)
                if loss_type == 'cross_entropy':
                    loss_function = loss_fn(target_logits, data_.target.to(device))
                else:
                    target = data_.target
                    one_hot_targets = torch.eye(2)[target]
                    target = torch.tensor(one_hot_targets).float().to(device)
                    loss_function = loss_fn(target_logits, target)

                loss_line = criterion_label(line_logits, data_.y.to(device))
                
                loss = alpha * loss_function + beta * loss_line

                total_test_loss += [loss.item()*batch_size]

                target_logits = torch.sigmoid(target_logits)

                target_out = torch.argmax(target_logits,dim=1).float()
                
                generated_texts.append(target_out)
                groundtruth_texts.append(data_.target)

                line_pre = torch.sigmoid(line_logits)

                line_pre_out = torch.argmax(line_pre, dim=1).float()

                generated_line_texts.append(line_pre_out)
                groundtruth_line_texts.append(data_.y)

                filter_correct_function_predictions_logit_label(line_logits_per,line_labels_per,target_out,  data_.target,line_logits,data_.y,data_.batch, all = False, saved = False) 

                filter_correct_function_predictions_logit_label(line_logits_per_all,line_labels_per_all,target_out,  data_.target,line_logits,data_.y,data_.batch, all = True, saved = False) 

        result_line = evaluate_code_line(line_logits_per,line_labels_per, line_logits_per_all,line_labels_per_all)

        val = evaluate_code_only_coarse(generated_texts,groundtruth_texts,dataset_name, dev_type='test')

        save_generated_data('/root/autodl-tmp/result/16/key_test_data.json', generated_texts, groundtruth_texts, line_logits_per_all, line_labels_per_all)

        logging.info(f'epoch: {epoch} , test loss: {np.sum(total_test_loss)}')
        logging.info("function level result")
        for key in sorted(val.keys()):
            logging.info("  %s = %s", key, str(round(val[key], 6)))
        logging.info("line level classification result")
        for key in sorted(val_line.keys()):
            logging.info("  %s = %s", key, str(round(val_line[key], 6)))
        logging.info("line level rank result")
        for key in sorted(result_line.keys()):
            value = result_line[key]
            if isinstance(value, list):
                logging.info("  %s = %s", key, str(value))
            else:
                logging.info("  %s = %s", key, str(round(result_line[key], 6)))
        
import math

def calculate_top_k_precent(line_labels_per,sort_ids,top_k):
    nums = 0
    top_k_recall = 0
    for idx in range(len(sort_ids)): 
        if sum(line_labels_per[idx]) == 0:
            nums+=1
            continue
        k = math.ceil(top_k * len(sort_ids[idx]))

        for num in sort_ids[idx][:k]:
            if line_labels_per[idx][num] == 1:
                top_k_recall+=1
                break
    if (len(line_labels_per)-nums)==0:
        return 0
    return round(top_k_recall/(len(line_labels_per)),6)


def calculate_top_k(line_labels_per,sort_ids,top_k):
    nums = 0
    top_k_recall = 0
    for idx in range(len(sort_ids)):
        if sum(line_labels_per[idx]) == 0:
            nums+=1
            continue
        for num in sort_ids[idx][:top_k]:
            if line_labels_per[idx][num] == 1:
                top_k_recall+=1
                break
    if (len(line_labels_per)-nums)==0:
        return 0
    # print(nums)
    return round(top_k_recall/(len(line_labels_per)),6)

def calculate_ifa(line_labels_per,sort_ids):
    ifas = []
    for idx in range(len(sort_ids)): 
        for id, num in enumerate(sort_ids[idx]):
            if line_labels_per[idx][num] == 1:
                ifas.append(id)
                break
    
    ifa_mean = np.mean(ifas)
    ifa_median = np.median(ifas)

    if len(ifas) == 0:
        ifa_max = -1
        ifa_min = -1
    else:
        ifa_max = np.max(ifas)
        ifa_min = np.min(ifas)
    return ifa_min,ifa_max,ifa_median,ifa_mean


def filter_correct_function_predictions_logit_label(line_logits_per,line_labels_per,target_out, target,line_logits,line_labels,data_batch, all = False, saved = False):
    
    line_logits = line_logits[:,1]
    if all :
        correct_function_indices = (target == 1)
    else:
        correct_function_indices = (target_out == target) & (target == 1)
    
    for i in range(correct_function_indices.size(0)):
        if correct_function_indices[i]: 
            batch_indices = (data_batch == i).nonzero(as_tuple=True)[0]
            line_logits_per.append(line_logits[batch_indices])
            line_labels_per.append(line_labels[batch_indices])

def save_generated_data(file_path, generated_texts, groundtruth_texts, line_logits_per_all, line_labels_per_all):

    data_to_save = {
        'generated_texts': torch.cat(generated_texts).tolist(),
        'groundtruth_texts': torch.cat(groundtruth_texts).tolist(),
        'line_logits_per_all': [logit.tolist() for logit in line_logits_per_all],
        'line_labels_per_all': [label.tolist() for label in line_labels_per_all] 
    }
    
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)

def filter_correct_function_predictions(target_out, target,line_logits,line_labels,data_batch, all = False, saved = False):
    
    line_logits = line_logits[:,1]

    if all :
        correct_function_indices = (target == 1)
    else:
        correct_function_indices = (target_out == target) & (target == 1)
    total_correct_functions = correct_function_indices.sum().item()
    
    selected_line_logits = []
    selected_line_labels = []
    
    for i in range(correct_function_indices.size(0)):
        if correct_function_indices[i]: 
            batch_indices = (data_batch == i).nonzero(as_tuple=True)[0]
            selected_line_logits.append(line_logits[batch_indices])
            selected_line_labels.append(line_labels[batch_indices])
    
    top1 = 0
    top3  = 0
    top5 = 0
    top10 = 0
    ifa = []
    for i in range(total_correct_functions):
        logits = selected_line_logits[i]
        labels = selected_line_labels[i]
          
        top1 += top_k_accuracy_line_level(logits, labels, 1)
        top3 += top_k_accuracy_line_level(logits, labels, 3)
        top5 += top_k_accuracy_line_level(logits, labels, 5)
        top10 += top_k_accuracy_line_level(logits, labels, 10)
        
        ifa.append(ifa_line_level(logits, labels))

    return top1, top3, top5, top10, ifa, total_correct_functions

def ifa_line_level(logits, labels):
    if logits.size(0) == 0:
        return 0.0

    sorted_indices = torch.argsort(logits, descending=True)
    ifa = -1
    for i in range(sorted_indices.size(0)):
        if labels[sorted_indices[i]] == 1:
            ifa = i
            break
    
    return ifa

def top_k_accuracy_line_level(line_logits, line_labels, k):
    if line_logits.size(0) == 0:
        return 0.0

    if k > line_logits.size(0):
        k = line_logits.size(0)
    top_k_preds = torch.topk(line_logits, k, dim=0).indices  
    correct = 0
    for i in range(top_k_preds.size(0)):
        if line_labels[top_k_preds[i]] == 1:
            correct = 1
            break

    return correct 



def evaluate_code_line(line_logits_per,line_labels_per, line_logits_per_all,line_labels_per_all):
    top_k_percent = [0.05,0.10,0.2]
    top_k_fun = [1,2,3,4,5,6,7,8,9,10]
    total_nums = len(line_labels_per)
    sort_ids = []
    if line_logits_per:
        sort_id = [torch.argsort(logits, descending=True) for logits in line_logits_per]
        sort_ids.extend(sort_id)
    
    top_k_recall_percent = []
    for k in top_k_percent:
        top_k_recall_percent.append(calculate_top_k_precent(line_labels_per,sort_ids,k))

    
    top_k_recall_fun = []
    for k in top_k_fun:
        top_k_recall_fun.append(calculate_top_k(line_labels_per,sort_ids,k))

    sort_ids_all = []
    if line_logits_per_all:
        sort_id = [torch.argsort(logits, descending=True) for logits in line_logits_per_all]
        sort_ids_all.extend(sort_id)

    top_k_recall_percent_all = []
    for k in top_k_percent:
        top_k_recall_percent_all.append(calculate_top_k_precent(line_labels_per_all,sort_ids_all,k))

    top_k_recall_fun_all = []
    for k in top_k_fun:
        top_k_recall_fun_all.append(calculate_top_k(line_labels_per_all,sort_ids_all,k))

    ifa_min,ifa_max,ifa_median,ifa_mean = calculate_ifa(line_labels_per_all,sort_ids_all)

    result_line = {

        "top_k_percent":top_k_percent, # 
        "top_k_recall_percent_all": top_k_recall_percent_all,
        "top_k_fun": top_k_fun,
        "top_k_all": top_k_recall_fun_all,
        "top_k_recall_percent": top_k_recall_percent,
        "top_k":top_k_recall_fun,
        "vul_nums":len(line_labels_per_all),
        "ifa_min":ifa_min,
        "ifa_max":ifa_max,
        "ifa_median":ifa_median,
        "ifa_mean":ifa_mean
    }
    return result_line


def evaluate_code_only_coarse(out,label,dataset_name, dev_type='val', good_value=0, bad_value=1):
    model_utils = Utilities()
    return model_utils.get_index_by_coarse_label(out, label)
