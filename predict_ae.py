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

from net.ae import *

# Device configuration
device = torch.device("cuda")


def generate_bgl(name, window_size, step):
    num_sessions = 0
    inputs = []
    predict = []
    print('dataset at:')
    print('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name)
    with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line[:window_size])
            predict.append(line[-1])
            num_sessions += 1

    print('Number of sessions({}): {}'.format(name, num_sessions))
    return inputs, predict


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='ae', choices=['vae', 'ae'])
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=16, type=int)

    parser.add_argument('-window_size', default=20, type=int)
    parser.add_argument('-dataset', type=str, default='bgl', choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=20, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-dropout', default=0.0, type=float)

    parser.add_argument('-step', default=10, type=int)
    parser.add_argument('-error_threshold', default=2, type=float)
    
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_epochs = args.epoch
    latent_length = args.latent_length
    dropout = args.dropout
    threshold = args.error_threshold
    threshold = torch.tensor(threshold)
    input_size = 1

    log = 'model/' + \
    'dataset=' + args.dataset + \
    '_window_size=' + str(window_size) + \
    '_hidden_size=' + str(hidden_size) + \
    '_latent_length=' + str(latent_length) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs) + \
    '_dropout=' + str(dropout)

    log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
    log = log + '_' + args.model + '.pt' 
    print('retrieve model from: ', log)

    criterion = nn.CrossEntropyLoss()

    if args.dataset == 'bgl':
        dataset, predict = generate_bgl('dataset.txt', window_size, args.step)
        num_classes = 377

    len_dataset = len(dataset)


    if args.model == 'ae':
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

    TP = 0
    FP = 0
    tbar = tqdm(dataset)
    with torch.no_grad():
        pre_loss = [0.0, 0.0, 0.0]
        for index, line in enumerate(tbar):
            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(line, dtype=torch.long).to(device)
            label = label.unsqueeze(0)

            output = model(seq)
            output = output.permute(0,2,1)
            loss = criterion(output, label).data
            if sum(pre_loss) > 6 and predict[index] == -1:
                FP += 1
            # elif loss > pre_loss[-1] + threshold and predict[index] == -1:
            #     FP += 1
            # elif loss > pre_loss[-1] + threshold and predict[index] == -2:
            #     TP += 1
            elif sum(pre_loss) > 6 and predict[index] == -2:
                TP += 1
            for i in range(len(pre_loss)-1):
                pre_loss[i] = pre_loss[i+1]
            pre_loss[-1] = loss

            tbar.set_description('FP: %.3f TP: %.3f' % (FP, TP))



    # Compute precision, recall and F1-measure
    len_abnormal = 46141
    FN = len_abnormal - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
    