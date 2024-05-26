import warnings
import pandas as pd
import numpy as np
from dpo1 import construct_dataset
# 改了
class CARTreeNode():
    def __init__(self, feature, split_spot, mse,  samples, hit=-1):
        self.feature = feature
        self.split_spot = split_spot
        self.mse = mse
        self.samples = samples
        self.hit = hit # 指是否到叶节点，默认为-1；否则会显示为震级的大小
        self.left = None
        self.right = None
# 原先是ginisplitting，改了
def MSESplitting(dataset, label):
    min_mse = -1
    min_feature = None  # 添加默认值
    split_spot = None  # 添加默认值
    # 按照数据集的特征遍历
    for feature in dataset.columns:
        # 遍历某个特征下所有可能的值，注意，每种值只遍历一次，而不是一行一行遍历
        if len(dataset[feature].value_counts().index) > 1:
            for i in dataset[feature].value_counts().index:
                left_label = label[dataset[feature] <= i]
                right_label = label[dataset[feature] > i]
                # print(left_label.shape, right_label.shape)
                mse = 0
                if left_label.shape[0] == 0:
                    mse += np.sum((right_label['magnitude'] - np.mean(right_label['magnitude'])) ** 2)
                elif right_label.shape[0] == 0:
                    mse += np.sum((left_label['magnitude'] - np.mean(left_label['magnitude'])) ** 2) 
                else:
                    left_label_mean, right_label_mean = np.mean(left_label['magnitude']), np.mean(right_label['magnitude'])
                    mse += np.sum((left_label['magnitude'] - left_label_mean) ** 2)
                    mse += np.sum((right_label['magnitude'] - right_label_mean) ** 2)
                    if mse < min_mse or min_mse == -1:
                        min_mse = mse
                        min_feature = feature
                        split_spot = i
            else:
                continue
        # print("min_feature:", min_feature, "split_spot:", split_spot, "min_mse:", min_mse)
    return min_feature, split_spot, min_mse
        
def TestCart(cart, label, dataset):
    error = 0.00
    for i, r in dataset.iterrows():
        # print(curNode.feature, curNode.split_spot, curNode.hit)
        curNode = cart
        while curNode.hit == -1:
            # print(curNode.feature, r[curNode.feature], curNode.split_spot, curNode.hit)
            if r[curNode.feature] <= curNode.split_spot:
                curNode = curNode.left
            elif r[curNode.feature] > curNode.split_spot:
                curNode = curNode.right
        error += np.abs(curNode.hit - label['magnitude'][i]) / label['magnitude'][i]
        print("The prediction is ", curNode.hit, "The real value is ", label['magnitude'][i])
    return error / 100

def PredictResult(cart, sample):
    curNode = cart
    while curNode.hit == -1:
        if sample[curNode.feature] <= curNode.split_spot:
            curNode = curNode.left
        else:
            curNode = curNode.right
    return curNode.hit

def ConstructCART(dataset, label, CurNode):
    # 终点条件：样本数小于等于1，或者样本的震级只有一种
    if CurNode.samples <= 10 or len(label['magnitude'].value_counts().index) == 1:
    # if CurNode.samples <= 1 or CurNode.mse <= 0.1:
        CurNode.hit = np.mean(label['magnitude'])
        print("<end of this branch, the prediction is ", CurNode.hit, ">")
        return 
        
    elif CurNode.feature is None:
        # 一开始的空节点，需要先计算一次切分点
        min_feature, split_spot, min_mse = MSESplitting(dataset, label)
        CurNode.feature = min_feature
        CurNode.split_spot = split_spot
        CurNode.mse = min_mse
        CurNode.samples = dataset.shape[0]
        # 调试
        print("The root of this tree:", CurNode.feature, CurNode.samples, CurNode.split_spot)
        ConstructCART(dataset, label, CurNode)
    else:
        # 先对数据集进行切分
        # print("Current node:", CurNode.feature, CurNode.samples, CurNode.split_spot)
        left_dataset, right_dataset = dataset[dataset[CurNode.feature] <= CurNode.split_spot], dataset[dataset[CurNode.feature] > CurNode.split_spot]
        left_label, right_label = label[dataset[CurNode.feature] <= CurNode.split_spot], label[dataset[CurNode.feature] > CurNode.split_spot]
        # 同时计算左右节点的最佳切分点
        l_min_feature, l_split_spot, l_min_mse = MSESplitting(left_dataset, left_label)
        r_min_feature, r_split_spot, r_min_mse = MSESplitting(right_dataset, right_label)
        l_node = CARTreeNode(l_min_feature, l_split_spot, l_min_mse, left_dataset.shape[0])
        r_node = CARTreeNode(r_min_feature, r_split_spot, r_min_mse, right_dataset.shape[0])
        CurNode.left = l_node
        CurNode.right = r_node
        # 调试
        print(l_node.feature, ": ", l_node.samples,"left mse:", l_node.mse, r_node.feature, ":",  r_node.samples, "right mse:", r_node.mse)
        ConstructCART(left_dataset, left_label, l_node)
        ConstructCART(right_dataset, right_label, r_node)
        
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    raw = pd.read_csv('earthquake_data.csv', sep=',', na_filter=None)
    dataset_x, dataset_label = construct_dataset(raw)
    Cart = CARTreeNode(None, None, float('inf'), dataset_x.shape[0])
    # 每次递归会输出该节点的左右节点的特征和样本数，到叶节点时给出决策。
    # 递归采取左侧深度优先
    ConstructCART(dataset_x, dataset_label, Cart)
    print("the error is", TestCart(Cart, dataset_label, dataset_x), "%")
    
        