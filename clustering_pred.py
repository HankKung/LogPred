import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
from tqdm import tqdm
import os
import numpy as np
from net.ae import AE, KMEANS
from net.vae import VRAE
import random
from deeplog.model import *


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_bgl(name, window_size):
    num_sessions = 0
    inputs = set()
    with open('bgl/window_'+str(window_size)+'future_0/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_'+str(window_size)+'future_0/' + name, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.add(line)
            num_sessions += 1

    print('Number of sessions({}): {}'.format(name, num_sessions))
    return inputs

def generate_hdfs(name, window_size):
    hdfs = set()
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            ### pad 28 if the sequence len is less than the window_size (log key start from 0 to 27)
            line = line + [28] * (window_size + 1 - len(line))
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                hdfs.add(tuple(seq))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs

def generate_random_hdfs(window_size, num_samples):
    hdfs = []
    for i in range(num_samples):
        line = [random.randint(0, 28) for j in range(window_size)]
        hdfs.append(line)
    return hdfs


if __name__ == '__main__':

    # Hyperparameters
    batch_size = 2048
    input_size = 1
    model_dir = 'model'
    
    parser = argparse.ArgumentParser()
    # ae
    parser.add_argument('-model', type=str, default='ae', choices=['ae', 'vae', 'dl'])
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=20, type=int)
    parser.add_argument('-window_size', default=20, type=int)
    parser.add_argument('-dropout', default=0.0, type=float)

    # training
    parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=150, type=int)
    parser.add_argument('-lr', default=0.001, type=float)

    # k-means
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-threshold', default=0.1, type=float)

    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    latent_length = args.latent_length
    window_size = args.window_size
    num_epochs = args.epoch
    dropout = args.dropout
    k = args.k
    threshold = args.threshold

    if args.dataset == 'hd':
        train_normal_loader = generate_hdfs('hdfs_train', window_size)
        test_normal_loader = generate_hdfs('hdfs_test_normal', window_size)
        test_abnormal_loader = generate_hdfs('hdfs_test_abnormal', window_size)
        num_classes = 28
        if args.model != 'dl':
            num_classes +=1
    elif args.dataset == 'bgl':
        test_normal_loader = generate_bgl('normal_test.txt', window_size)
        test_abnormal_loader = generate_bgl('abnormal_test.txt', window_size)
        num_classes = 1848
    
    len_train_normal = len(train_normal_loader)
    len_normal = len(test_normal_loader)
    len_abnormal = len(test_abnormal_loader)

    model_path = 'model/'
    if args.model == 'ae' or args.model == 'vae':
        log = model_path + \
        'dataset=' + args.dataset + \
        '_window_size=' + str(window_size) + \
        '_hidden_size=' + str(hidden_size) + \
        '_latent_length=' + str(latent_length) + \
        '_num_layer=' + str(num_layers) + \
        '_epoch=' + str(num_epochs) + \
        '_dropout=' + str(dropout) 
        log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
        log = log + '_' + args.model + '.pt' 
    else:
        log = 'model/num_layer=' + str(num_layers) + \
        '_window_size=' + str(window_size) + \
        '_hidden=' + str(hidden_size) + \
        '_dataset=' + args.dataset +\
        '_epoch='+str(args.epoch)
        log = log + '_' + args.model
        log = log + '.pt'
    print('retrieve model from: ', log)


    if args.model == 'ae':
        model = AE(input_size, hidden_size, latent_length, num_layers, num_classes, window_size)
    elif args.model == 'vae':
        model = VRAE(sequence_length=window_size,
            number_of_features=1,
            num_classes=num_classes,
            hidden_size=hidden_size,
            latent_length=latent_length,
            training=False)
    elif args.model == 'dl':
        model = DL(input_size, hidden_size, num_layers, num_classes)

    model = model.to(device)
    model.load_state_dict(torch.load(log))
    model.eval()

    k_means_path = log[:-3] + '_' + str(k) + '/'

    # normal_embedded 

    clusters = []
    for i in range(k):
        cluster = np.load(k_means_path + 'center_' + str(i) + '.npy')
        cluster = torch.from_numpy(cluster).cuda()
        # print(cluster.data)
        clusters.append(cluster)

    FP = 0
    tbar = tqdm(train_normal_loader)
    with torch.no_grad():
        normal_min_dist = 0.0
        for index, line in enumerate(tbar):
            line = list(line)
            line[-1] = random.randint(0, 28)
            line = tuple(line)
            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            latent = model.get_latent(seq)
            min_dist = 100.0
            for i, cluster in enumerate(clusters):

                dist = torch.sqrt(torch.sum(torch.mul(latent - cluster, latent - cluster)))
                min_dist = dist.item() if dist.item() < min_dist else min_dist
            if min_dist > threshold:
                FP += 1
            normal_min_dist += min_dist
            tbar.set_description('train normal min dist: %.3f' % (normal_min_dist / (index + 1)))
    print('accuracy:')
    print(FP/len_train_normal)

    TP = 0
    FP = 0
    # Test the model
    tbar = tqdm(test_normal_loader)
    with torch.no_grad():
        normal_min_dist = 0.0

        for index, line in enumerate(tbar):
            line = list(line)
            line[-1] = random.randint(0, 28)
            line = tuple(line)
            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            latent = model.get_latent(seq)
            min_dist = 100.0
            for i, cluster in enumerate(clusters):
                dist = torch.sqrt(torch.sum(torch.mul(latent - cluster, latent - cluster)))
                min_dist = dist.item() if dist.item() < min_dist else min_dist
            if min_dist > threshold:
                FP += 1
            normal_min_dist += min_dist
            tbar.set_description('normal min dist: %.3f' % (normal_min_dist / (index + 1)))

    tbar = tqdm(test_abnormal_loader)
    with torch.no_grad():
        abnormal_min_dist = 0.0
        for index, line in enumerate(tbar):
            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            latent = model.get_latent(seq)
            min_dist = 100.0
            for i, cluster in enumerate(clusters):
                dist = torch.sqrt(torch.sum(torch.mul(latent - cluster, latent - cluster)))
                min_dist = dist.item() if dist.item() < min_dist else min_dist
            if min_dist > threshold:
                TP += 1
            abnormal_min_dist += min_dist
            tbar.set_description('abnormal min dist: %.3f' % (abnormal_min_dist / (index + 1)))

    print('normal_avg_dist:')
    print(normal_min_dist/len_normal)
    print('abnormal_avg_dist:')
    print(abnormal_min_dist/len_abnormal)

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')


    # generate random sequence
    random_hdfs = generate_random_hdfs(window_size, 10000)
    # test random seq
    avg_dist = 0.0
    with torch.no_grad():
        for index, line in enumerate(random_hdfs):
            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            latent = model.get_latent(seq)
            min_dist = 100.0
            for i, cluster in enumerate(clusters):
                dist = torch.sqrt(torch.sum(torch.mul(latent - cluster, latent - cluster)))
                min_dist = dist.item() if dist.item() < min_dist else min_dist
            # print('random seq: ', line, '~~min_distance: ', min_dist)
            avg_dist += min_dist
        print('average dist ', avg_dist/10000)

