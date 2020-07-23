import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from tqdm import tqdm
from vae.vae import VRAE
import os
import random

# Device configuration
device = torch.device("cuda")


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


    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='vae', choices=['vae', 'ae'])
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=20, type=int)

    parser.add_argument('-window_size', default=20, type=int)
    parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=150, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-dropout', default=0.0, type=float)

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
    log = log + '_' + args.model + args.caption + '.pt' 
    print('retrieve model from: ', log)

    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'hd':
        random_loader = generate_random_hdfs(window_size, 10000)
        train_loader = generate_hdfs('hdfs_train', window_size)
        test_normal_loader = generate_hdfs('hdfs_test_normal', window_size)
        test_abnormal_loader = generate_hdfs('hdfs_test_abnormal', window_size)
        num_classes = 28
        num_classes +=1
    elif args.dataset == 'bgl':
        # train_loader = generate_bgl('normal_train.txt', window_size)
        test_normal_loader = generate_bgl('normal_test.txt', window_size)
        test_abnormal_loader = generate_bgl('abnormal_test.txt', window_size)
        num_classes = 1848
    # len_random = len(random_loader)
    # len_train = len(train_loader)
    len_normal = len(test_normal_loader)
    len_abnormal = len(test_abnormal_loader)

    if args.model == 'vae'
        model = VRAE(sequence_length=window_size,
                number_of_features = 1,
                num_classes = num_classes,
                hidden_size = hidden_size,
                latent_length = latent_length,
                dropout_rate=dropout)
    elif args.model == 'ae':
        model = AE(input_size, hidden_size, latent_length, num_layers, num_classes, window_size, dropout_rate=dropout)
    
    model = model.to(device)
    model.load_state_dict(torch.load(log))
    model.eval()

    # random_loss = 0
    # tbar = tqdm(random_loader)
    # with torch.no_grad():
    #     normal_error = 0.0
    #     for index, line in enumerate(tbar):
    #         # print(line)
    #         seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
    #         label = torch.tensor(line).to(device)
    #         output, _ = model(seq)
    #         output = output.permute(0,2,1)
    #         label = label.unsqueeze(0)
    #         loss = criterion(output, label)

    #         random_loss +=loss.item()
    #     print('random_loss: ', random_loss/len_random)

    # train_FP = 0
    # train_loss = 0
    # tbar = tqdm(train_loader)
    # with torch.no_grad():
    #     for index, line in enumerate(tbar):
    #         # print(line)
    #         line = list(line)
    #         random.shuffle(line)

    #         line = tuple(line)
    #         seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
    #         label = torch.tensor(line).to(device)
    #         output, _ = model(seq)
    #         output = output.permute(0,2,1)
    #         label = label.unsqueeze(0)
    #         loss = criterion(output, label)

    #         if loss.item() > threshold:
    #             train_FP += 1
    #         train_loss +=loss.item()
    #     print('train_accuracy: ', train_FP/len_train)
    #     print('train_loss: ', train_loss/len_train)

    # train_FP = 0
    # train_loss = 0
    # tbar = tqdm(train_loader)
    # with torch.no_grad():
    #     for index, line in enumerate(tbar):
    #         # print(line)
    #         seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
    #         label = torch.tensor(line).to(device)
    #         output, _ = model(seq)
    #         output = output.permute(0,2,1)
    #         label = label.unsqueeze(0)
    #         loss = criterion(output, label)

    #         if loss.item() > threshold:
    #             train_FP += 1
    #         train_loss += loss.item()
    #     print('train_accuracy:', train_FP/len_train)
    #     print('train_loss: ', train_loss/len_train)


    TP = 0
    FP = 0
    # Test the model
    tbar = tqdm(test_normal_loader)
    with torch.no_grad():
        normal_error = 0.0
        for index, line in enumerate(tbar):
            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(line).to(device)
            if args.model == 'vae':
                output, _ = model(seq)
            else:
                output = model(seq)
            output = output.permute(0,2,1)
            label = label.unsqueeze(0)
            loss = criterion(output, label)
            if loss.item() > threshold:
                FP += 1
            normal_error +=loss.item()
            tbar.set_description('normal error: %.3f' % (normal_error / (index + 1)))

    tbar = tqdm(test_abnormal_loader)
    with torch.no_grad():
        abnormal_error = 0.0
        for index, line in enumerate(tbar):

            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(line).to(device)
            if args.model == 'vae':
                output, _ = model(seq)
            else:
                output = model(seq)
            output = output.permute(0,2,1)
            label = label.unsqueeze(0)
            loss = criterion(output, label)

            if loss.item() > threshold:
                TP += 1
            abnormal_error += loss.item()
            tbar.set_description('abnormal error: %.3f' % (abnormal_error / (index + 1)))

    print('normal_avg_error:')
    print(normal_error/len_normal)
    print('abnormal_avg_error:')
    print(abnormal_error/len_abnormal)

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')