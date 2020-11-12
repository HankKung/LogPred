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


def generate_bgl(name, window_size, step, slide):
    num_sessions = 0
    inputs = []
    print('dataset at:')
    print('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'slide_'+str(slide)  + 'remove_8/' + name)
    with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'slide_'+str(slide)+ 'remove_8/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'slide_'+str(slide)+ 'remove_8/' + name, 'r') as f:
        for line in f.readlines():         

            line = list(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line)


    print('Number of sessions({}): {}'.format(name, num_sessions))
    return inputs


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='ae', choices=['ae'])
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=16, type=int)

    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-dataset', type=str, default='bgl', choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=30, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-dropout', default=0.0, type=float)

    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_epochs = args.epoch
    latent_length = args.latent_length
    dropout = args.dropout

    input_size = 1

    log = 'model/' + \
    'dataset=' + args.dataset + \
    '_window_size=' + str(window_size) + \
    '_hidden_size=' + str(hidden_size) + \
    '_latent_length=' + str(latent_length) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs) + \
    '_dropout=' + str(dropout) + \
    '_step=0' 

    log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
    log = log + '_' + args.model + '.pt' 
    print('retrieve model from: ', log)

    criterion = nn.CrossEntropyLoss()
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

    Slide = [5]
    Step = [3]
    for slide in Slide:
        for step in Step:
            dataset = generate_bgl('dataset.txt', window_size, step, slide)

            
            tbar = tqdm(dataset)
            loss_set = []
            # label_set = []
            with torch.no_grad():
                for index, line in enumerate(tbar):
                    loss_line = []
                    log_line = []
                    for i in range(len(line) - window_size+1):
                        seq = torch.tensor(line[i:i+window_size], dtype=torch.float).view(-1, window_size, input_size).to(device)
                        label = torch.tensor(line[i:i+window_size], dtype=torch.long).to(device)
                        label = label.unsqueeze(0)

                        output = model(seq)
                        output = output.permute(0,2,1)
                        loss = criterion(output, label).item()

                        loss_line.append(loss)

                    log_line = line
                    loss_set.append((loss_line, log_line))

            normal_loss = []
            abnormal_loss = []
            random_dataset = []

            for i, (line, log) in enumerate(loss_set):
                for j in range(len(line) - slide + 1):
                    loss_seq = line[j:j+slide]
                    log_seq = log[j:j+window_size+slide-1]
                    if j < len(line) - (slide + step) + 1:
                        loss_seq.append(0)
                        loss_seq.append(log_seq)
                        normal_loss.append(loss_seq)
                    else:
                        loss_seq.append(1)
                        loss_seq.append(log_seq)
                        abnormal_loss.append(loss_seq)
                    # random_dataset.append(loss_seq)

            len_data = len(normal_loss)
            random.shuffle(normal_loss)
            train_set = normal_loss[:int(len_data * 0.8)]
            val_set = normal_loss[int(len_data * 0.8):]
            print('slide: ' , str(slide), 'step: ', str(step))
            print('train_len:')
            print(len(train_set))
            print('val_normal:')
            print(len(val_set))
            print('abnormal:')
            print(len(abnormal_loss))

            data_dir = 'bgl/loss_'+'future_' + str(step) + 'slide_' + str(slide)
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)
            with open(data_dir+'/normal_loss_train_set.txt', 'w') as f:
                for i, line in enumerate(train_set):
                    for item in line:
                        f.write(str(item) + ' ')
                    f.write('\n')
            with open(data_dir+'/normal_loss_val_set.txt', 'w') as f:
                for i, line in enumerate(val_set):
                    for item in line:
                        f.write(str(item) + ' ')
                    f.write('\n')

            with open(data_dir+'/abnormal_loss_val_set.txt', 'w') as f:
                for i, line in enumerate(abnormal_loss):
                    for item in line:
                        f.write(str(item) + ' ')
                    f.write('\n')






            