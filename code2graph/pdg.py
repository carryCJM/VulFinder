import json

def load_cpg_data(node_file, edge_file):
    """加载 CPG 节点和边数据"""
    with open(node_file, 'r') as nf:
        nodes = json.load(nf)
    with open(edge_file, 'r') as ef:
        edges = json.load(ef)
    return nodes, edges


def create_pdg(nodes, edges):
    """
    根据节点和边信息创建程序依赖图 (PDG)，将同一行代码的节点合并。
    返回包含合并后节点、边和行号的信息。
    """

    output = {
        "nodes": [],
        "node_code": [],
        "node_info": [],
        "edges": [],
        "edge_type": []
    }

    node_line_map = {}
    for node in nodes:
        if 'lineNumber' in node:
            node_line_map[node['id']] = node['lineNumber']
            output['nodes'].append(node['lineNumber'])
            output['node_code'].append(node['code'])
            output['node_info'].append("")  # 添加必要的节点信息

    for edge in edges:
        src_line = node_line_map.get(edge[0])
        tgt_line = node_line_map.get(edge[1])
        if src_line is not None and tgt_line is not None:
            output['edges'].append([src_line, tgt_line])
            output['edge_type'].append(edge[2])  # 保留边的类型信息

    # 找到控制依赖
    # 在cfg中找到控制依赖
    control_dependencies = []
    for edge in edges:
        if edge[2] == 'control':  # 假设边的类型信息中 'control' 表示控制依赖
            src_line = node_line_map.get(edge[0])
            tgt_line = node_line_map.get(edge[1])
            if src_line is not None and tgt_line is not None:
                control_dependencies.append((src_line, tgt_line))
    
    output['control_dependencies'] = control_dependencies


    # 找到数据依赖
    data_dependencies = []
    for edge in edges:
        if edge[2] == 'data':  # 假设边的类型信息中 'data' 表示数据依赖
            src_line = node_line_map.get(edge[0])
            tgt_line = node_line_map.get(edge[1])
            if src_line is not None and tgt_line is not None:
                data_dependencies.append((src_line, tgt_line))
    
    output['data_dependencies'] = data_dependencies

    return output

def save_pdg_to_jsonl(output, output_file):
    # """
    # 将 PDG 节点和边的信息保存为 JSONL 格式
    # """
    # with open(output_file, 'w') as out_file:
    #     for line_num, node_data in pdg_nodes.items():
    #         # 获取边信息
    #         edges = pdg_edges.get(line_num, [])
    #         node_data['edges'] = edges
            
    #         # 将数据写入文件
    #         json_line = json.dumps(node_data)
    #         out_file.write(json_line + "\n")
    """将输出数据保存到 JSONL 文件中"""
    with open(output_file, 'w') as f:
        f.write(json.dumps(output) + "\n")


def merge_nodes(nodes):
    merged_nodes = {}
    for node in nodes:
        line_number = node.get('lineNumber')
        if line_number:
            if line_number not in merged_nodes:
                merged_nodes[line_number] = {
                    "code": node.get('code', ''),
                    "info": set()  # 使用集合避免重复
                }
            # 简化信息，仅添加关键元素
            if 'variable' in node:
                merged_nodes[line_number]['info'].add(node['variable'])

    # 转换 info 集合为列表
    for line, data in merged_nodes.items():
        data['info'] = list(data['info'])

    return merged_nodes

def process_edges(edges, node_line_map):
    processed_edges = []
    edge_types = {'CFG', 'REACHING_DEF', 'CALL', 'REF'}  # 保留关键的边类型
    for edge in edges:
        src_line = node_line_map.get(edge[0])
        tgt_line = node_line_map.get(edge[1])

        # 跳过自循环边
        if src_line == tgt_line:
            continue
        
        if [src_line, tgt_line, edge[2]] in processed_edges:
            # print("重复的边",src_line, tgt_line, edge[2])
            continue
        # 保留关键的边类型
        if src_line and tgt_line and edge[2] in edge_types:
            processed_edges.append([src_line, tgt_line, edge[2]])
    print(processed_edges)
    return processed_edges


def compute_reaching_definitions(cpg, pdg):
    reaching_defs = {}
    for node in pdg.nodes():
        reaching_defs[node] = set()
        for pred in pdg.predecessors(node):
            if pdg[pred][node].get('type') == 'REACHES':
                reaching_defs[node].add(pred)
    return reaching_defs


def main(node_file, edge_file, output_file):
    # 1. 加载 CPG 数据
    nodes, edges = load_cpg_data(node_file, edge_file)

    merged_nodes = merge_nodes(nodes)
    # 创建一个从原始节点ID映射到行号的映射
    node_line_map = {node['id']: node['lineNumber'] for node in nodes if 'lineNumber' in node}
    
    # 过滤边
    processed_edges = process_edges(edges, node_line_map)

    # 构建最终输出格式
    output = {
        "nodes": list(merged_nodes.keys()),
        "node_code": [node['code'] for node in merged_nodes.values()],
        "node_info": [node['info'] for node in merged_nodes.values()],
        "edges": [[edge[0], edge[1]] for edge in processed_edges],
        "edge_type": [edge[2] for edge in processed_edges]
    }
    
    # 2. 创建 PDG
    output = create_pdg(nodes, edges)
    
    # 3. 保存 PDG 到 JSONL
    save_pdg_to_jsonl(output, output_file)


if __name__ == "__main__":
    node_file = r'code2graph\joern\test_pdg_new\scpy2-bad1.c_nodes.json'
    edge_file = r'code2graph\joern\test_pdg_new\scpy2-bad1.c_edges.json'
    output_file = r'code2graph\joern\test_pdg_new\scpy2-bad1.c_pdg.json'
    
    main(node_file, edge_file, output_file)
