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
    # num_sessions = 0
    # inputs = set()
    # outputs = []
    # num_keys = set()
    # with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f_len:
    #     file_len = len(f_len.readlines())
    # with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f:
    #     for line in f.readlines():
    #         num_sessions += 1
    #         source = tuple(map(lambda n: n, map(float, line.strip().split())))
    #         # label = tuple(map(lambda n: n, map(int, line.strip().split()[-1])))
    #         inputs.add(source)
            # outputs.append(label)
    new_inputs = []
    new_outputs = []
    num_normal = 0
    for i, label in enumerate(outputs):
        # print(label)
        if label == (0,) and num_normal < 10000:
            new_inputs.append(inputs[i])
            new_outputs.append(label)
            num_normal += 1
        elif label == (1,):
            new_inputs.append(inputs[i])
            new_outputs.append(label)
        # break
    # inputs = list(inputs)
    # outputs = list(outputs)
    # new_inputs = []
    # new_outputs = []
    # for line in inputs:
    #     # print(line)
    #     new_inputs.append(line[:window_size])
    #     new_outputs.append(int(line[-1]))


    # new_normal_inputs = set()
    # new_abnormal_inputs = []
    # new_abnormal_outputs = []
    # num_noraml = 0
    # for i, label in enumerate(outputs):
    #     # print(label)
    #     if label == (0,):
    #         new_normal_inputs.add(inputs[i])
    #         # new_normal_outputs.append(label)
    #         num_noraml += 1
    #     elif label == (1,):
    #         new_abnormal_inputs.append(inputs[i])
    #         new_abnormal_outputs.append(label)

    # new_normal_inputs = list(new_normal_inputs)
    # new_inputs = new_normal_inputs + new_abnormal_inputs
    # len_normal = len(new_normal_inputs)
    # new_outputs = [(0,) for _ in range(len_normal)]
    # new_outputs = new_outputs + new_abnormal_outputs
    print(name)
    print('Number of sessions: {}'.format(len(new_inputs)))
    dataset = TensorDataset(torch.tensor(new_inputs, dtype=torch.float), torch.tensor(new_outputs))
    return dataset, len(new_inputs), len(new_normal_inputs)
    # return dataset

def retrain_generate_bgl(name, window_size, step):
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
    # num_sessions = 0
    # inputs = set()
    # outputs = []
    # num_keys = set()
    # with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f_len:
    #     file_len = len(f_len.readlines())
    # with open('bgl/window_'+str(window_size)+'future_' + str(step) + 'remove_8/' + name, 'r') as f:
    #     for line in f.readlines():
    #         num_sessions += 1
    #         source = tuple(map(lambda n: n, map(float, line.strip().split())))
    #         # label = tuple(map(lambda n: n, map(int, line.strip().split()[-1])))
    #         inputs.add(source)
            # outputs.append(label)
    new_normal_inputs = set()
    new_abnormal_inputs = []
    new_abnormal_outputs = []
    num_noraml = 0
    for i, label in enumerate(outputs):
        # print(label)
        if label == (0,):
            new_normal_inputs.add(inputs[i])
            # new_normal_outputs.append(label)
            num_noraml += 1
        elif label == (1,):
            new_abnormal_inputs.append(inputs[i])
            new_abnormal_outputs.append(label)

    new_normal_inputs = list(new_normal_inputs)
    new_inputs = new_normal_inputs + new_abnormal_inputs
    len_normal = len(new_normal_inputs)
    new_outputs = [(0,) for _ in range(len_normal)]
    new_outputs = new_outputs + new_abnormal_outputs
        # break
    # inputs = list(inputs)
    # outputs = list(outputs)
    # new_inputs = []
    # new_outputs = []
    # for line in inputs:
    #     # print(line)
    #     new_inputs.append(line[:window_size])
    #     new_outputs.append(int(line[-1]))
    print(name)
    print('Number of sessions: {}'.format(len(inputs)))
    dataset = TensorDataset(torch.tensor(new_inputs, dtype=torch.float), torch.tensor(new_outputs))
    return dataset

if __name__ == '__main__':

    # Hyperparameters
    batch_size = 512
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
    parser.add_argument('-retrain_epoch', type=int, default=0)
    parser.add_argument('-resume', type=str, default='')
    parser.add_argument('-r_lr', default=0.000005, type=float)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size

    num_epochs = args.epoch if args.retrain_epoch == 0 else args.retrain_epoch
    dropout = args.dropout

    if args.retrain_epoch != 0:
        seq_dataset = retrain_generate_bgl('train_loss_set.txt', window_size, args.step)
    else:
        seq_dataset, data_n, normal_data_n = generate_bgl('train_loss_set.txt', window_size, args.step)
    # val_dataset = generate_bgl('abnormal_test.txt', window_size)
    num_classes = 2
    
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    

    log = 'dataset='+ str(args.dataset) + \
    '_window_size=' + str(window_size) + \
    '_hidden_size=' + str(hidden_size) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs) + \
    '_dropout=' + str(dropout) 
    log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
    log = log + '_' + 'lstm'
    print('store model at:')
    print(log)
    writer = SummaryWriter(log_dir='log/' + log)


    model = LSTM_predict(input_size, hidden_size, num_layers)
    model = model.to(device)
    if args.resume != '':
        model.load_state_dict(torch.load(args.resume))

    torch.FloatTensor([0.1, 2])
    nSamples = [data_n, data_n-normal_data_n]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights[0] = normedWeights[0]
    normedWeights = torch.FloatTensor(normedWeights).cuda()

    # Loss and optimizer
    if args.dataset == 'bgl_loss':
        criterion = torch.nn.MSELoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss(normedWeights).cuda()

    if args.retrain_epoch != 0:
        lr = args.r_lr
    else:
        lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Train the model
    total_step = len(dataloader)
    for epoch in range(num_epochs): 
        train_loss = 0.0
        tbar = tqdm(dataloader)
        model.train()
        for step, (seq, label) in enumerate(tbar):
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            label = torch.tensor(label).view(-1).to(device)
            # print(label.shape)
            output = model(seq)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            tbar.set_description('Train loss: %.3f' % (train_loss / (step + 1)))
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    if args.retrain_epoch != 0:
        log = log + '_retrain'
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')