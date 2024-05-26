import pandas as pd
from dpo1 import construct_dataset
from CART1 import CARTreeNode, ConstructCART, TestCart,PredictResult
from os import name
import warnings
from sklearn.model_selection import train_test_split

import numpy as np

def random_subset(dataset_x, dataset_label, subset_size=0.6):
    indices = np.arange(len(dataset_x))
    random_indices = np.random.choice(indices, size=int(len(dataset_x) * subset_size), replace=False)
    return dataset_x.iloc[random_indices], dataset_label.iloc[random_indices]


def select_three_features():
    # 定义所有的特征
    features = ['time', 'tsunami', 'cdi', 'mmi', 'depth', 'latitude', 'longitude', 'location']
    # 从所有特征中随机选择三个
    selected_features = np.random.choice(features, 5, replace=False)
    return selected_features.tolist()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    raw = pd.read_csv('earthquake_data.csv', sep=',', na_filter=None)
    dataset_x, dataset_label = construct_dataset(raw)
    
    # 将数据集切分为训练集和测试集
    train_x, test_x, train_label, test_label = train_test_split(dataset_x, dataset_label, test_size=0.1, random_state=42)
    

    # 构建随机森林
    num_trees = 13  # 设置随机森林中树的数量
    for_tree_features = []
    forest = []
    for _ in range(num_trees):
        # 随机抽取训练样本的一部分

        train_x_subset, train_label_subset = random_subset(train_x, train_label)
        
        # 随机抽取特征
        selected_features = select_three_features()
        for_tree_features.append(selected_features)


        #数据集只保留选中的特征的数据
        train_x_subset = train_x_subset[selected_features]
        for_tree_features.append(selected_features)
        # 构建决策树
        tree = CARTreeNode(None, None, 1, train_x_subset.shape[0])

        ConstructCART(train_x_subset, train_label_subset, tree)
        forest.append(tree)
    '''
    # 测试随机森林的准确率
    correct = 0
    for i in range(len(test_x)):
        count = 0
        for j in range(num_trees):
            tree = forest[j]
            selected_features = for_tree_features[j]
            if PredictResult(tree, test_x.iloc[i]):
                count += 1
        if count > num_trees / 2:
            prediction = 1
        else:
            prediction = 0
        if prediction == test_label['tsunami'].iloc[i]:
            correct += 1
    accuracy = correct / len(test_x)
    print("Random Forest accuracy:", accuracy)
    '''
    correct = 0
    error=0
    for i in range(len(test_x)):
        count = 0
        for j in range(num_trees):
            tree = forest[j]
            selected_features = for_tree_features[j]
            count +=PredictResult(tree, test_x.iloc[i])
        count = count / num_trees
        error += np.abs(count - test_label['magnitude'].iloc[i]) / test_label['magnitude'].iloc[i]
        
    average_error = error / len(test_x)
    print("Random Forest average Error:", average_error)