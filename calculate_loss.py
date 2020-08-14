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
    print('dataset at:')
    print('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name)
    with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f:
        for line in f.readlines():         

            line = list(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line)
            # for i in range(len(line) - window_size + 1):
            #     inputs.append(line[i:window_size])
            #     if i == len(line)-step or i < len(line)-step:
            #         predict.append(0)
            #     else:
            #         predict.append(1)


            #     num_sessions += 1

    print('Number of sessions({}): {}'.format(name, num_sessions))
    return inputs


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='ae', choices=['vae', 'ae'])
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=20, type=int)

    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-dataset', type=str, default='bgl', choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=150, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-dropout', default=0.0, type=float)

    parser.add_argument('-step', default=10, type=int)
    parser.add_argument('-slide', default=10, type=int)
    parser.add_argument('-error_threshold', default=0.1, type=float)
    
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_epochs = args.epoch
    latent_length = args.latent_length
    dropout = args.dropout
    threshold = args.error_threshold

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

    dataset = generate_bgl('dataset.txt', window_size, args.step)
    num_classes = 377

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
    tbar = tqdm(dataset)
    loss_set = []
    # label_set = []
    with torch.no_grad():
        for index, line in enumerate(tbar):
            loss_line = []
            label_line = []
            for i in range(len(line) - window_size +1):
                seq = torch.tensor(line[i:i+window_size], dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(line[i:i+window_size], dtype=torch.long).to(device)
                label = label.unsqueeze(0)

                output = model(seq)
                output = output.permute(0,2,1)
                loss = criterion(output, label).item()

                loss_line.append(loss)
                # if i < len(line) - (window_size + step) + 1:
                #     label_line.append(0)
                # else:
                #     label_line.append(1)
            loss_set.append(loss_line)
            # label_set.append(label_line)
        
    random_dataset = []
    for i, line in enumerate(loss_set):
        print(len(line))
        for j in range(len(line) - args.slide + 1):
            loss_seq = line[j:j+window_size]
            if j < len(line) - (window_size + args.step) + 1:
                loss_seq.append(0)
            else:
                loss_seq.append(1)
            random_dataset.append(loss_seq)

    len_data = len(random_dataset)
    random.shuffle(random_dataset)
    train_set = random_dataset[:int(len_data * 0.8)]
    val_set = random_dataset[int(len_data * 0.8):]

    train_ab = 0
    for line in train_set:
        if line[-1] == 1:
            train_ab +=1
    val_ab = 0
    for line in val_set:
        if line[-1] == 1:
            val_ab += 1

    print('training set len:')
    print(len(train_set))
    print('abnormal in training set:')
    print(train_ab)
    print('val set len:')
    print(len(val_set))
    print('abnormal in val set:')
    print(val_ab)


    data_dir = 'bgl/window_'+str(window_size)+'future_' + str(args.step) + 'remove_8/'
    with open(data_dir+'train_loss_set.txt', 'w') as f:
        for i, line in enumerate(train_set):
            for item in line:
                f.write(str(item) + ' ')
            f.write('\n')

    with open(data_dir+'val_loss_set.txt', 'w') as f:
        for i, line in enumerate(val_set):
            for item in line:
                f.write(str(item) + ' ')
            f.write('\n')



    