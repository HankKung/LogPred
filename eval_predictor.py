import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
from tqdm import tqdm
import os
import random
from net.loss_lstm import *

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_bgl(name, window_size, step):
    num_sessions = 0
    inputs = []
    outputs = []
    num_keys = set()
    with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            source = tuple(map(lambda n: n, map(float, line.strip().split()[:-1])))
            label = tuple(map(lambda n: n, map(int, line.strip().split()[-1])))
            inputs.append(source)
            outputs.append(label)

    # with open('bgl/loss_window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + 'train_loss_set.txt', 'r') as f:
    #     for line in f.readlines():
    #         num_sessions += 1
    #         source = tuple(map(lambda n: n, map(float, line.strip().split()[:-1])))
    #         label = tuple(map(lambda n: n, map(int, line.strip().split()[-1])))
    #         inputs.append(source)
    #         outputs.append(label)

    num_abnoraml = 0
    for i, label in enumerate(outputs):
        # print(label)
        if label == (1,):
            num_abnoraml +=1

    print('Number of sessions: {}'.format(num_sessions))
    return inputs, outputs, num_abnoraml


if __name__ == '__main__':

    # Hyperparameters
    batch_size = 1
    input_size = 1
    model_dir = 'model'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=1, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-dataset', type=str, default='loss_bgl')
    parser.add_argument('-epoch', default=100, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-step', default=5, type=int)
    parser.add_argument('-dropout', default=0.0, type=float)
    parser.add_argument('-retrain', type=str, default='')
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_epochs = args.epoch
    dropout = args.dropout


    inputs, outputs, num_abnoraml = generate_bgl('val_loss_set.txt', window_size, args.step)
    # val_dataset = generate_bgl('abnormal_test.txt', window_size)
    num_classes = 2
    
    
    model_dir = 'model/'
    log = model_dir + 'dataset='+ str(args.dataset) + \
    '_window_size=' + str(window_size) + \
    '_hidden_size=' + str(hidden_size) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs) + \
    '_dropout=' + str(dropout) 
    log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
    log = log + '_' + 'lstm'
    if args.retrain != '':
        log = log + '_retrain'
    log = log +'.pt' 
    print('retrieve model from: ', log)

    model = LSTM_predict(input_size, hidden_size, num_layers)
    model = model.to(device)

    model.load_state_dict(torch.load(log))
    model.eval()
    # Loss and optimizer


    TP = 0
    FP = 0
    TN = 0
    TP_loss = []
    FP_loss = []
    TN_loss = []
    FN_loss = []

    store_loss = []
    tbar = tqdm(inputs)
    with torch.no_grad():
        for index, line in enumerate(tbar):

            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = outputs[index]
            output = model(seq)
            _, pred = torch.max(output, 1)
            if label == (0,) and pred.data == 1:
                FP += 1
                FP_loss.append(line)
                store_loss.append(list(line)+list(label))
            elif label == (0,) and pred.data == 0:
                TN += 1
                TN_loss.append(line)
            elif label == (1,) and pred.data == 1:
                TP += 1
                TP_loss.append(line)
            elif label == (1,) and pred.data == 0:
                FN_loss.append(line)
            tbar.set_description('FP: %.3f TP: %.3f' % (FP, TP))


    # len_data = len(store_loss)
    # random.shuffle(store_loss)
    # train_set = store_loss[:int(len_data * 0.8)]
    # val_set = store_loss[int(len_data * 0.8):]

    # TP_loss = np.array(TP_loss, dtype=np.float16)
    # FP_loss = np.array(FP_loss, dtype=np.float16)
    # TN_loss = np.array(TN_loss, dtype=np.float16)
    # FN_loss = np.array(FN_loss, dtype=np.float16)
    # np.save('TP_loss', TP_loss)
    # np.save('FP_loss', FP_loss)
    # np.save('TN_loss', TN_loss)
    # np.save('FN_loss', FN_loss)

    # data_dir = 'bgl/window_'+str(window_size)+'future_' + str(args.step) + 'remove_8/'
    # with open(data_dir+'train_remove_FN_loss_set.txt', 'w') as f:
    #     for i, line in enumerate(train_set):
    #         for item in line:
    #             f.write(str(item) + ' ')
    #         f.write('\n')

    # with open(data_dir+'val_remove_FN_loss_set.txt', 'w') as f:
    #     for i, line in enumerate(val_set):
    #         for item in line:
    #             f.write(str(item) + ' ')
    #         f.write('\n')

    FN = num_abnoraml - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('TP: {}, TN: {}'.format(TP, TN))

    print('Finished Predicting')



