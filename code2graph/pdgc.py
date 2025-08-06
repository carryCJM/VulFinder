import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from joern import run_joern
from pdg import main as generate_pdg
etype_onehot = {
    'CDG': 0,
    # 'REACHING_DEF': 1,
    'DDG': 1,
    'CALL': 2,
    
}

# 搭配多线程使用的
def process_file(file_path):
    # 生成 PDGC 和节点(sliced 的方法，暂时不用了)
    # pdgc, nodes = generate_pdgc(file_path)

    pdg = generate_pdg(file_path) # 这里，把节点排序输出了
    # 对node进行排序
    node_lines = sorted(list(pdg.nodes))
    # 获取节点代码
    # node_code = [pdg.nodes[data]['code'] for data in node_lines]

    # 遍历边节点
    edge = [[e[0], e[1]] for e in pdg.edges]
    edge_type = [etype_onehot[pdg.edges[e]['etype']] for e in pdg.edges]

    # 获取文件名
    file_name = file_path.split('\\')[-1]

    # 读取和更新 info.json 文件
    info_file_path = file_path + '.info.json'
    with open(info_file_path, 'r') as f:
        info = json.load(f)

    info['file_name'] = file_name
    info['nodes'] = node_lines
    # info['node_code'] = node_code
    info['edges'] = edge
    info['edge_type'] = edge_type

    with open(info_file_path, 'w') as f:
        json.dump(info, f, indent=4)

    # print(f'Processed {file_path}')

# 主程序
def Tread(files_list_path):
    # 传入的是一个文件列表，里面是要处理的文件的路径
    files = []
    # 首先要读取文件列表，把文件路径读取出来，再一个个执行
    with open(files_list_path, 'r') as f:
        for file_path in f.readlines():
            files.append(file_path.strip())

    with ThreadPoolExecutor() as executor:
        # 使用 tqdm 显示进度条
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files}
        for future in tqdm(as_completed(future_to_file), total=len(files)):
            file_path = future_to_file[future]
            try:
                future.result()  # 获取结果以处理异常
            except Exception as e:
                print(f'Error processing {file_path}: {e}')


# 遍历文件夹,保存所有文件的相对地址,到txt文件中,适用于val,和test
def save_files_in_folder(folder_path,filelist_path):
    with open(filelist_path, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            # 如果是json bin csv文件，就跳过
            for file in files:
                if file.endswith('.json') or file.endswith('.bin') or file.endswith('.csv'):
                    continue
                file_path = os.path.join(root, file)
                f.write(file_path + '\n')


from joern import run_joern_batch

# 批量解析文件夹中的文件
def batch_parse_c2cpg(file_path):
    run_joern_batch(file_path)
    

# 针对train,多线程 调用 batch_parse_c2cpg
def thread_c2cpg(file_path_list):
    # 传入的是,要批量处理的列表
    with ThreadPoolExecutor() as executor:
        # 使用 tqdm 显示进度条
        future_to_file = {executor.submit(batch_parse_c2cpg, file_path): file_path for file_path in file_path_list}
        for future in tqdm(as_completed(future_to_file), total=len(file_path_list)):
            file_path = future_to_file[future]
            try:
                future.result()  # 获取结果以处理异常
            except Exception as e:
                print(f'Error processing {file_path}: {e}')


import time

if __name__=='__main__':
    # 记录时间
    starttime=time.time()

    # 保存文件列表到txt文件中,便于后续批量处理,使用test 和val,,train,先保存下来,后边批处理
    save_files_in_folder(file_path, filelist_path) 

    # 从config.json中读取数据集路径
    with open('./config.json', 'r') as f:
        config = json.load(f)

    # 主要就是先把csv文件中 process_data的数据读取出来，保存到单独的文件中，将project作为文件夹的名称, 将index+file_name的最后一个/后的内容，作为文件名。
    # 批量 val
    file_path = config['processed_data_path_val'] # 输出目录,val数据集的输出目录
    filelist_path = config['processed_data_list_val'] # 文件列表目录
    
    # 批量train
    file_path = config['processed_data_path_train'] # 输出目录,test数据集的输出目录
    filelist_path = config['processed_data_list_test'] # 文件列表目录

    file_path = config['processed_data_path_test'] # 输出目录,val数据集的输出目录
    filelist_path = config['processed_data_list_test'] # 文件列表目录
    

    # 批量,先将代码转cpg,保存下来
    batch_parse_c2cpg(filelist_path)


    # 优化成多线程来处理,将cpg转换为pdgc
    for i in range(0,8):
        filelist_path = config['processed_data_list_train_'+str(i)]
        Tread(filelist_path)

    Tread(filelist_path)
    

    print(time.time()-starttime)
