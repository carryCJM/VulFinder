import argparse
from run_utils import train_graph_adapter
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graph adapter')
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='bigvul_all', choices=['arxiv', 'instagram', 'reddit','sliced_mini','sliced','bigvul','bigvul_mini','bigvul_big'])
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--llm_shape', type=int, default=4096)
    parser.add_argument('--hiddensize_gnn', type=int, default=1024)
    parser.add_argument('--hiddensize_fusion', type=int, default=1024)
    parser.add_argument('--normal_hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_ratio', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2) 
    parser.add_argument('--warm_up_ratio', type=float, default=0.1) 
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_size_eval', type=int, default=512)
    parser.add_argument('--batch_size_test', type=int, default=512)
    
    parser.add_argument('--pos_weight_set', type=float, default=-1) 
    parser.add_argument('--gamma', type=str, default="cross")
    parser.add_argument('--use_GNN', type=bool, default=True)
    parser.add_argument('--GNN_type', type=str, default='ImprovedGNNModel') 
    parser.add_argument('--plm_path', type=str, default='/root/autodl-model/CodeLlama-7b-hf',
                                   help='path of plm')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)

    args = parser.parse_args()
  
    start_time = time.time()
    train_graph_adapter(args) 
    train_time = time.time()-start_time
    print("Training time: ", train_time)
  

