import os
import json

# 合并处理后的 PDG 数据
def merge_pdgc_data(folder_path,output_path):
    # 如果输出文件不存在，则创建一个空文件
    # 获取文件所在的目录路径
    directory = os.path.dirname(output_path)

    # 检查目录是否存在，如果不存在则创建它
    if not os.path.exists(directory):
        os.makedirs(directory)

    total_count = 0  # 初始化数据计数
    with open(output_path, 'w') as outfile:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('info.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # 假设每个 info.json 存储一个对象或一个对象列表
                        if isinstance(data, list):
                            total_count += len(data)  # 如果是列表，累加元素数量
                            for item in data:
                                json.dump(item, outfile)
                                outfile.write('\n')  # 添加换行符
                        else:
                            total_count += 1  # 单个对象
                            json.dump(data, outfile)
                            outfile.write('\n')  # 添加换行符

    print(f"总共有 {total_count} 个数据。")  # 输出统计信息


if __name__ == '__main__':
    # # Usage example
    folder_path = r'processed_data_val'
    output_path = 'datasets/bigvul/merged_pdgc_val.jsonl'
    merge_pdgc_data(folder_path, output_path)

    folder_path = r'processed_data'
    output_path = 'datasets/bigvul/merged_pdgc_test.jsonl'
    merge_pdgc_data(folder_path, output_path)

    folder_path = r'processed_data_train'
    output_path = 'datasets/bigvul/merged_pdgc_train.jsonl'
    merge_pdgc_data(folder_path, output_path)