'''
重新写一个可以根据自己设定的数来提取训练集的数据处理程序
这个版本只有训练集
'''

import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import argparse
import collections
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser("IP")

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_of_segment', type=int, default=6, help='Divide the data processing process into several stages')
parser.add_argument('--windows', type=int, default=15, help='patche size')
parser.add_argument('--percentage', type=int, default=5, help='percentage of trainingsets')
parser.add_argument('--rand_seed', type=int, default=1, help='random seed')
args = parser.parse_args()


def seed_torch(seed):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_torch(args.rand_seed)


def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length #X坐标轴
        assign_1 = value % Col + pad_length #y坐标轴
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    # print(matrix.shape)
    selected_rows = matrix[:, range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, :, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch

percentage = args.percentage

sample_200 = [15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 15] #460

if percentage == 10:
    sample_200 = [15, 100, 70, 50, 50, 50, 12, 50, 15, 70, 100, 50, 50, 100, 50, 15]  # 844
elif percentage == 20:
    sample_200 = [18, 200, 100, 75, 75, 75, 12, 75, 15, 100, 200, 75, 50, 200, 75, 25]  # 1367
elif percentage == 30:
    sample_200 = [25, 400, 200, 75, 150, 200, 15, 150, 15, 300, 600, 180, 60, 350, 120, 25]  # 1367
elif percentage == 40:
    sample_200 = [25, 500, 280, 75, 175, 250, 15, 175, 15, 380, 800, 180, 70, 500, 135, 30]  # 1367
elif percentage == 50:
    sample_200 = [30, 624, 365, 81, 204, 327, 18, 200, 18, 436, 1127, 209, 80, 1032, 150, 34]  # 1367
SAMPLE = sample_200



def rSampling(groundTruth, sample_num=SAMPLE):  # divide dataset into train and test datasets
    whole_loc = {}
    train = {}
    val = {}
    m = np.max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        whole_loc[i] = indices
        train[i] = indices[:sample_num[i]]
        val[i] = indices[sample_num[i]:]

    whole_indices = []
    train_indices = []
    val_indices = []
    for i in range(m):
        whole_indices += whole_loc[i]
        train_indices += train[i]
        val_indices += val[i]
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
    return whole_indices, train_indices, val_indices

def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    new_matrix = np.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)),
                            'constant', constant_values=0)
    return new_matrix


def split_indices(data,n):
    total_length = len(data)
    each_length = total_length//n

    splited_length = []
    for i in range(n):
        if i !=n-1:
            splited_length.append(each_length)
        elif i == n-1:
            splited_length.append(total_length-each_length*(i))

    return splited_length

def countEachClassInTrain(y_count_train,num_class):
    each_class_num=np.zeros([num_class])
    for i in y_count_train:
        i=int(i)
        each_class_num[i]=each_class_num[i]+1
    return each_class_num

mat_data = sio.loadmat('Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
mat_gt = sio.loadmat('indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']
print(data_IN.shape)

bands = data_IN.shape[-1]
nb_classes = np.max(gt_IN)

new_gt_IN = gt_IN


INPUT_DIMENSION_CONV = bands
INPUT_DIMENSION = bands


TOTAL_SIZE = np.sum(gt_IN>0)

TRAIN_SIZE = sum(SAMPLE)
print("")

TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE

img_channels = bands

PATCH_LENGTH = int(args.windows/2)  # Patch_size 9*9

MAX = data_IN.max()

data_IN = np.transpose(data_IN, (2, 0, 1))

data_IN = data_IN - np.mean(data_IN, axis=(1, 2), keepdims=True)
data_IN = data_IN / MAX

data = data_IN.reshape(np.prod(data_IN.shape[:1]), np.prod(data_IN.shape[1:]))

gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])

padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

CATEGORY = nb_classes

all_indices, train_indices, test_indices = rSampling(gt)

y_train = gt[train_indices] - 1
y_test = gt[test_indices] - 1
print('训练集长度(按照label核算)：{}'.format(len(y_train)))
print('测试集长度(按照label核算)：{}'.format(len(y_test)))

print('训练集长度(按照data核算)：{}'.format(len(train_indices)))
print('测试集长度(按照data核算)：{}'.format(len(test_indices)))
splited_train_len = split_indices(train_indices,args.num_of_segment)
print("训练集分段情况：")
print(splited_train_len)
splited_test_len = split_indices(test_indices,args.num_of_segment)
print("测试集分段情况：")
print(splited_test_len)

X_train=np.empty([0,bands,args.windows,args.windows])
X_test=np.empty([0,bands,args.windows,args.windows])

for i in range(args.num_of_segment):
    print("---------分割第{}段---------".format(i + 1))
    X_train_i = np.empty([splited_train_len[i], bands, args.windows, args.windows])
    X_test_i = np.empty([splited_test_len[i], bands, args.windows, args.windows])

    print("start_index:{}".format(i*splited_train_len[1]))
    print("end_index:{}".format(i*splited_train_len[1]+splited_train_len[i]-1))

    train_assign = indexToAssignment(train_indices[i*splited_train_len[1]:i*splited_train_len[1]+splited_train_len[i]-1],
                                     PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])

    for j in range(len(train_assign)):
        X_train_i[j] = selectNeighboringPatch(padded_data, PATCH_LENGTH, train_assign[j][0], train_assign[j][1])

    test_assign = indexToAssignment(test_indices[i*splited_test_len[1]:i*splited_test_len[1]+splited_test_len[i]-1],
                                    PATCH_LENGTH, whole_data.shape[1], whole_data.shape[2])
    for j in range(len(test_assign)):
        X_test_i[j] = selectNeighboringPatch(padded_data, PATCH_LENGTH, test_assign[j][0], test_assign[j][1])

    X_train=np.vstack((X_train,X_train_i))
    X_test=np.vstack((X_test,X_test_i))


def savePreprocessedData(percentage, X_trainPatches, X_testPatches, y_trainPatches, y_testPatches,
                        windowSize):
    with open(str(percentage)+"XtrainWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, X_trainPatches)
    with open(str(percentage)+"XtestWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, X_testPatches)
    with open(str(percentage)+"ytrainWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, y_trainPatches)
    with open(str(percentage)+"ytestWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
        np.save(outfile, y_testPatches)


print("train集各类别的数量：")
print(countEachClassInTrain(y_train,nb_classes))
print("test集各类别的数量：")
print(countEachClassInTrain(y_test,nb_classes))

savePreprocessedData(percentage, X_train, X_test, y_train, y_test, windowSize=args.windows)
