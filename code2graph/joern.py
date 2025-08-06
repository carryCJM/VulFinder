
import logging
import os
import subprocess
import sys
from difflib import SequenceMatcher
import numpy as np
import warnings
sys.path.append(os.path.dirname(__file__))
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

# constants
helpers_dir = os.path.dirname(__file__) # 获取helpers目录,即SlicedLocator-main/helpers
root_dir = os.path.dirname(helpers_dir) # 获取根目录,即SlicedLocator-main
logs_dir = os.path.join(root_dir, 'logs')
html_dir = os.path.join(root_dir, 'html')
os.makedirs(logs_dir, exist_ok=True)

# print(f'helpers_dir: {helpers_dir}')
# print(f'root_dir: {root_dir}')
# print(f'logs_dir: {logs_dir}')
# print(f'html_dir: {html_dir}')

logging.basicConfig(filename=os.path.join(logs_dir, 'joern.log'),
                    filemode='a',
                    level=logging.NOTSET,
                    format='%(asctime)s - %(levelname)s - %(process)d - %(funcName)s - %(message)s')


def run_joern_batch(filepaths):
    """Run Joern on a batch of files

    Args:
        filepaths (list): list of filepaths
    """
    scala_script = '../../../helper/gen_graph_batch.scala'
    
    command = f'.\joern --script {scala_script} --param """filepaths={filepaths}"""'
    subprocess.run(command, shell=True)



def run_joern(filepath: str, scala_script : str ,i=0):
    """Use Joern to generate a graph for a given program

    Args:
        filepath (str): path to a .c file
    """
    assert os.path.isfile(filepath)  
    
    if not os.path.exists(filepath):
        print(f'{filepath}: FAIL. filepath does not exist.')
        logging.error(f'{filepath}: FAIL. filepath does not exist.')
        return False
    
    savedir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    
    if os.path.exists(os.path.join(savedir, f'{filename}_edges.json')) and \
       os.path.exists(os.path.join(savedir, f'{filename}_nodes.json')):
        # print(f'{filepath}: PASS. Already parsed by Joern.')
        logging.info(f'{filepath}: PASS. Already parsed by Joern.')
        return True

    # joern_2 里直接执行这个命令
    command = f'.\joern --script {scala_script} --param """filename={filepath}"""'

    subprocess.run(command, shell=True)
    
    if os.path.exists(os.path.join(savedir, f'{filename}_edges.json')) and \
       os.path.exists(os.path.join(savedir, f'{filename}_nodes.json')):
        print(f'{filepath}: SUCCESS.')
        logging.info(f'{filepath}: SUCCESS.')
        return True
    else:
        logging.error(f'{filepath}: FAIL. Joern error.')
        print(f'{filepath}: FAIL. Joern error.')
        return False


