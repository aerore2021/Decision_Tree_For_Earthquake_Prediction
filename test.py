import pandas as pd
from dpo1 import construct_dataset
from CART import CARTreeNode, ConstructCART, TestCart,PredictResult
from os import name
import warnings
from sklearn.model_selection import train_test_split

import numpy as np
'''调库测试数据集
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
'''
def select_three_features():
    # 定义所有的特征
    features = ['time', 'magnitude', 'cdi', 'mmi', 'depth', 'latitude', 'longitude', 'location']
    # 从所有特征中随机选择三个
    selected_features = np.random.choice(features, 5, replace=False)
    #添加一个'tsunami'特征
    #selected_features = np.append(selected_features, 'tsunami')
    return selected_features.tolist()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    raw = pd.read_csv('earthquake_data.csv', sep=',', na_filter=None)
    dataset_x, dataset_label = construct_dataset(raw)
    dataset_x, test_x, dataset_label, test_label = train_test_split(dataset_x, dataset_label, test_size=0.1, random_state=42)

    selected_features=select_three_features()
    print("selected features: ", selected_features)
    train_x_subset = dataset_x[selected_features]
    test_x = test_x[selected_features]
    Cart = CARTreeNode(None, None, 1, train_x_subset.shape[0])
    # 每次递归会输出该节点的左右节点的特征和样本数，到叶节点时给出决策。
    # 递归采取左侧深度优先
    ConstructCART(train_x_subset, dataset_label, Cart)
    print("the accuracy is ", TestCart(Cart, test_label, test_x))
    
    