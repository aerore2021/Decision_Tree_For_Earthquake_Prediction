import warnings
import pandas as pd
from dpo1 import construct_dataset

class CARTreeNode():
    def __init__(self, feature, split_spot, gini, samples, hit=-1):
        self.feature = feature
        self.split_spot = split_spot
        self.gini = float(gini)
        self.samples = samples
        self.hit = hit # 指是否到叶节点，默认为-1；如果会发生海啸为1，不发生为0.
        self.left = None
        self.right = None
        
def GiniSplitting(dataset, label):
    min_gini = float('inf')
    min_feature = None  # 添加默认值
    split_spot = None  # 添加默认值
    min_gini = 1
    # 按照数据集的特征遍历
    for feature in dataset.columns:
        # 遍历某个特征下所有可能的值，注意，每种值只遍历一次，而不是一行一行遍历
        for i in dataset[feature].value_counts().index:
            left = dataset[dataset[feature] <= i]
            right = dataset[dataset[feature] > i]
            gini = 0
            for data in [left, right]:
                if data.shape[0] == 0:
                    continue
                p = data[label['tsunami']==1].shape[0] / data.shape[0]
                sub_gini = 2 * p * (1 - p)
                gini += data.shape[0] / dataset.shape[0] * sub_gini
            if gini < min_gini:
                min_gini = gini
                min_feature = feature
                split_spot = i
    return min_feature, split_spot, min_gini
        
def TestCart(cart, label, dataset):
    score = 0
    curNode = cart
    for i, r in dataset.iterrows():
        if curNode.hit == -1:
            if r[curNode.feature] <= cart.split_spot:
                curNode = curNode.left
            elif r[curNode.feature] > cart.split_spot:
                curNode = curNode.right
        else:
            if curNode.hit == label['tsunami'][i]:
                score += 1
    return score / dataset.shape[0]

def PredictResult(cart, sample):
    curNode = cart
    while curNode.hit == -1:
        if sample[curNode.feature] <= curNode.split_spot:
            curNode = curNode.left
        else:
            curNode = curNode.right
    return bool(curNode.hit)

def ConstructCART(dataset, label, CurNode):
    # 终点条件
    if CurNode.samples <= 1 or CurNode.gini <= 0.1:
        CurNode.hit = 1 if label[label['tsunami'] == 1].shape[0] / label.shape[0] > 0.5 else 0
        print("<end of this branch, the decision is ", bool(CurNode.hit), ">")
        return 
        
    elif CurNode.feature is None:
        # 一开始的空节点，需要先计算一次切分点
        min_feature, split_spot, min_gini = GiniSplitting(dataset, label)
        CurNode.feature = min_feature
        CurNode.split_spot = split_spot
        CurNode.gini = min_gini
        CurNode.samples = dataset.shape[0]
        # 调试
        print("The root of this tree:", CurNode.feature, CurNode.samples)
        ConstructCART(dataset, label, CurNode)
    else:
        # 先对数据集进行切分
        left_dataset, right_dataset = dataset[dataset[CurNode.feature] <= CurNode.split_spot], dataset[dataset[CurNode.feature] > CurNode.split_spot]
        left_label, right_label = label[dataset[CurNode.feature] <= CurNode.split_spot], label[dataset[CurNode.feature] > CurNode.split_spot]
        # 同时计算左右节点的最佳切分点
        l_min_feature, l_split_spot, l_min_gini = GiniSplitting(left_dataset, left_label)
        r_min_feature, r_split_spot, r_min_gini = GiniSplitting(right_dataset, right_label)
        l_node = CARTreeNode(l_min_feature, l_split_spot, l_min_gini, left_dataset.shape[0])
        r_node = CARTreeNode(r_min_feature, r_split_spot, r_min_gini, right_dataset.shape[0])
        CurNode.left = l_node
        CurNode.right = r_node
        # 调试
        print(l_node.feature, ": ", l_node.samples, r_node.feature, ":",  r_node.samples)
        ConstructCART(left_dataset, left_label, l_node)
        ConstructCART(right_dataset, right_label, r_node)
        
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    raw = pd.read_csv('earthquake_data.csv', sep=',', na_filter=None)
    dataset_x, dataset_label = construct_dataset(raw)
    Cart = CARTreeNode(None, None, 1, dataset_x.shape[0])
    # 每次递归会输出该节点的左右节点的特征和样本数，到叶节点时给出决策。
    # 递归采取左侧深度优先
    ConstructCART(dataset_x, dataset_label, Cart)
    print("the accuracy is ", TestCart(Cart, dataset_label, dataset_x))
    
        