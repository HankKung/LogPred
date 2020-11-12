import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import time
import argparse
from tqdm import tqdm
from net.vae import VRAE
import os
import random
import json
import re
from net.ae import *

# Device configuration
device = torch.device("cuda")


def generate_bgl(name, window_size, step):
    num_sessions = 0
    inputs = []
    print('dataset at:')
    print('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name)
    with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.append((line,line))
            num_sessions += 1

    print('Number of sessions({}): {}'.format(name, num_sessions))
    return inputs


def generate_bgl_loss(name, step, slide):
    num_sessions = 0
    inputs = []
    print('dataset at:')
    print('bgl/loss_'+'future_' + str(step) + 'slide_' + str(slide) + '/' + name)
    with open('bgl/loss_'+'future_' + str(step) + 'slide_' + str(slide) + '/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/loss_'+'future_' + str(step) + 'slide_' + str(slide) + '/' + name, 'r') as f:
        for line in f.readlines():
            loss = tuple(map(lambda n: n, map(float, line.strip().split()[:slide])))
            log = tuple(map(lambda n: n, map(int, re.split(r",\s|\s|\[|\]|", line.strip())[slide+2:-1])))
            # print(re.split(r",\s|\s|\[|\]|", line.strip()))
            inputs.append((loss,log))
            num_sessions += 1

    print('Number of sessions({}): {}'.format(name, num_sessions))
    return list(inputs)

def generate_bgl_loss_full(window_size, step):
    num_sessions = 0
    inputs = set()
    print('dataset at:')
    print('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/normal_loss_val_set.txt')
    with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/normal_loss_val_set.txt') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/normal_loss_val_set.txt') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n, map(float, line.strip().split()[:-1])))
            inputs.add(line)
            num_sessions += 1
    with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/normal_loss_train_set.txt') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n, map(float, line.strip().split()[:-1])))
            inputs.add(line)
            num_sessions += 1
    return list(inputs)

def generate_hd(name, window_size):
# use set for testing, it will take about one min
# otherwise using list to store will take about 3 hrs.
    hdfs = set()
    # hdfs = []
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [28] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
            # hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


def generate_random_hdfs(window_size, num_samples):
    hdfs = []
    for i in range(num_samples):
        line = [random.randint(0, 28) for j in range(window_size)]
        hdfs.append(line)
    return hdfs

def store_log(dataset, name, log, step, slide):
    if dataset == 'bgl':
        data_dir = 'key_'
    else:
        data_dir = ''
    data_dir = data_dir + 'step_' + str(step) + 'slide_' +str(slide) + '/' 
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(data_dir + name + '_log.txt', 'w') as f:
        for i, item in enumerate(log):
            out = ''
            for j in item:
                out = out + str(j) + ' ' 
            f.write(out+'\n')

def store_loss(name, loss, step, slide):

    data_dir = 'step_' + str(step) + 'slide_' +str(slide) + '/' 
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(data_dir + name + '_loss.txt', 'w') as f:
        for i, item in enumerate(loss):
            out = ''
            for j in item:
                out = out + str(j) + ' ' 
            f.write(out+'\n')

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='ae', choices=['vae', 'ae'])
    parser.add_argument('-num_layers', default=1, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=16, type=int)

    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-dataset', type=str, default='bgl_loss', choices=['bgl', 'bgl_loss', 'bgl_loss_full'])
    parser.add_argument('-epoch', default=40, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-dropout', default=0.0, type=float)
    parser.add_argument('-step', default=5, type=int)
    parser.add_argument('-slide', default=3, type=int)
    parser.add_argument('-error_threshold', default=0.0005, type=float)
    
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size if args.slide == 0 else args.slide
    num_epochs = args.epoch
    latent_length = args.latent_length
    dropout = args.dropout
    threshold = args.error_threshold

    input_size = 1

    log = 'model/' + 'dataset=' + args.dataset

    log = log + '_window_size=' + str(window_size) if args.slide == 0 else log + '_slide=' + str(args.slide)
    log = log + '_hidden_size=' + str(hidden_size) + \
    '_latent_length=' + str(latent_length) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs) + \
    '_dropout=' + str(dropout) + \
    '_step=' +str(args.step)


    log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
    log = log + '_' + args.model + '.pt' 
    print('retrieve model from: ', log)

    if args.dataset == 'bgl_loss' or args.dataset == 'bgl_loss_full':
        criterion = torch.nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.dataset == 'hd':
        test_normal_loader = generate_hd('hdfs_test_normal', window_size)
        test_abnormal_loader = generate_hd('hdfs_test_abnormal', window_size)

        num_classes = 28
        num_classes +=1
    elif args.dataset == 'bgl':
        test_normal_loader = generate_bgl('normal_test.txt', window_size, args.step)
        test_abnormal_loader = generate_bgl('abnormal_test.txt', window_size, args.step)
        num_classes = 377
    elif 'bgl_loss' in args.dataset:
        test_normal_loader = generate_bgl_loss('normal_loss_val_set.txt', args.step, args.slide)
        test_abnormal_loader = generate_bgl_loss('abnormal_loss_val_set.txt', args.step, args.slide)
        num_classes = 1
    # elif args.dataset == 'bgl_loss_full':
    #     test_normal_loader = generate_bgl_loss_full(window_size, args.step)
    #     test_abnormal_loader = generate_bgl_loss('abnormal_loss_val_set.txt', window_size, args.step)
    #     num_classes = 1

    len_normal = len(test_normal_loader)
    len_abnormal = len(test_abnormal_loader)

    if args.model == 'vae':
        model = VRAE(sequence_length=window_size,
                number_of_features = input_size,
                num_classes = num_classes,
                hidden_size = hidden_size,
                latent_length = latent_length,
                dropout_rate=dropout)
    elif args.model == 'ae':
        model = AE(input_size,
                    hidden_size,
                    latent_length,
                    num_layers,
                    num_classes,
                    window_size,
                    dropout_rate=dropout)
    
    model = model.to(device)
    model.load_state_dict(torch.load(log))
    model.eval()

    if args.dataset == 'hd':
        TP = 0
        FP = 0

        # Test the model
        tbar = tqdm(test_normal_loader)
        with torch.no_grad():
            normal_error = 0.0
            FP_error = 0.0
            n = 0
            for index, seq in enumerate(tbar):
                seq = torch.tensor(seq, dtype=torch.float).to(device)
                label_full = torch.tensor(seq, dtype=torch.long).to(device)
                for i in range(seq.shape[0] - window_size):
                    inputs = seq[i:i + window_size]
                    label = label_full[i:i + window_size]

                    inputs = inputs.view(-1, window_size, input_size)
                    # label = torch.tensor(label).to(device).unsqueeze(0)
                    label = label.unsqueeze(0)

                    if args.model == 'vae':
                        output, _ = model(inputs)
                    else:
                        output = model(inputs)

                    output = output.permute(0,2,1)
                    # label = label.unsqueeze(0)

                    loss = criterion(output, label)
                    if loss.item() > threshold:
                        FP += 1
                        FP_error += loss.item()
                        break
                    else:
                        n += 1
                        normal_error +=loss.item()
                if n !=0 and FP !=0:
                    tbar.set_description('normal error: %.3f FP error: %.3f' % ((normal_error / n), (FP_error / (FP))))

        tbar = tqdm(test_abnormal_loader)
        with torch.no_grad():
            normal_error = 0.0
            TP_error = 0.0
            n = 0
            for index, seq in enumerate(tbar):
                seq = torch.tensor(seq, dtype=torch.float).to(device)
                label_full = torch.tensor(seq, dtype=torch.long).to(device)
                for i in range(len(seq) - window_size):
                    inputs = seq[i:i + window_size]
                    label = label_full[i:i + window_size]

                    inputs = inputs.view(-1, window_size, input_size)
                    label = label.unsqueeze(0)

                    if args.model == 'vae':
                        output, _ = model(inputs)
                    else:
                        output = model(inputs)
                        
                    output = output.permute(0,2,1)
                    loss = criterion(output, label)

                    if loss.item() > threshold:
                        TP += 1
                        TP_error += loss.item()
                        break
                    else:
                        normal_error += loss.item()
                        n += 1
                    if n !=0 and TP !=0:
                        tbar.set_description('normal error: %.3f TP error: %.2f' % ((normal_error / (n+ 1)), (TP_error / (TP + 1))))
    else:
        TP = 0
        FP = 0
        TP_set = []
        FP_set = []
        TN_set = []
        FN_set = []
        if args.dataset == 'bgl_loss':
            TP_loss = []
            FP_loss = []
            TN_loss = []
            FN_loss = []

        print(len(test_abnormal_loader))
        print(len(test_normal_loader))
        tbar = tqdm(test_normal_loader)
        with torch.no_grad():
            normal_error = 0.0
            for index, (line,log) in enumerate(tbar):
                seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
                
                if args.dataset == 'bgl_loss' or args.dataset == 'bgl_loss_full':
                    label = torch.tensor(line, dtype=torch.float).to(device)
                else:
                    label = torch.tensor(line, dtype=torch.long).to(device)
                label = label.unsqueeze(0)

                if args.model == 'vae':
                    output, _ = model(seq)
                else:
                    output = model(seq)
                    if args.dataset == 'bgl_loss' or args.dataset == 'bgl_loss_full':
                        label = label.unsqueeze(2)
                    else:
                        output = output.permute(0,2,1)
                loss = criterion(output, label)
                if loss.item() > threshold:
                    FP += 1
                    FP_set.append(log)
                    if args.dataset == 'bgl_loss':
                        FP_loss.append(line)
                else:
                    TN_set.append(log)
                    if args.dataset == 'bgl_loss':
                        TN_loss.append(line)

                normal_error +=loss.item()
                tbar.set_description('normal error: %.5f FP: %.1f' % (normal_error / (index + 1), FP))
        tbar = tqdm(test_abnormal_loader)
        with torch.no_grad():
            abnormal_error = 0.0
            for index, (line, log) in enumerate(tbar):
                seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)

                if args.dataset == 'bgl_loss' or args.dataset == 'bgl_loss_full':
                    label = torch.tensor(line, dtype=torch.float).to(device)
                else:
                    label = torch.tensor(line, dtype=torch.long).to(device)
                label = label.unsqueeze(0)

                if args.model == 'vae':
                    output, _ = model(seq)
                else:
                    output = model(seq)
                    if args.dataset == 'bgl_loss' or args.dataset == 'bgl_loss_full':
                        label = label.unsqueeze(2)
                    else:
                        output = output.permute(0,2,1)
                loss = criterion(output, label)

                if loss.item() > threshold:
                    TP += 1
                    TP_set.append(log)
                    if args.dataset == 'bgl_loss':
                        TP_loss.append(line)
                else:
                    FN_set.append(log)
                    if args.dataset == 'bgl_loss':
                        FN_loss.append(line)

                abnormal_error += loss.item()
                tbar.set_description('abnormal error: %.5f, TP: %.1f' % (abnormal_error / (index + 1), TP))

    store_log(args.dataset, 'TP', TP_set, args.step, window_size)
    store_log(args.dataset, 'FP', FP_set, args.step, window_size)
    store_log(args.dataset, 'TN', TN_set, args.step, window_size)
    store_log(args.dataset, 'FN', FN_set, args.step, window_size)

    if args.dataset == 'bgl_loss':
        store_loss('TP', TP_loss, args.step, window_size)
        store_loss('FP', FP_loss, args.step, window_size)
        store_loss('TN', TN_loss, args.step, window_size)
        store_loss('FN', FN_loss, args.step, window_size)
    # print('normal_avg_error:')
    # print(normal_error/len_normal)
    # print('abnormal_avg_error:')
    # print(abnormal_error/len_abnormal)

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    TN = len(test_normal_loader) - FP

    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('TP: {}, TN: {}'.format(TP, TN))
    print('Finished Predicting')
    