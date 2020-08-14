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
from net.dl import *

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_hd(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset

def generate_bgl(name, window_size):
    num_sessions = 0
    inputs = []
    outputs = []
    num_keys = set()
    with open(name+'/window_' + str(window_size) +'future_0remove_8//normal_train.txt', 'r') as f_len:
        file_len = len(f_len.readlines())
    with open(name+'/window_'+ str(window_size) + 'future_0remove_8/normal_train.txt', 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line[:window_size])
            outputs.append(line[-1])
            for key in line:
                num_keys.add(key)


    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('number of keys:{}'.format(len(num_keys)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


if __name__ == '__main__':

    # Hyperparameters
    batch_size = 2048
    input_size = 1
    model_dir = 'model'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-model', type=str, default='dl', choices=['dl', 'att', 'trainable_att'])
    parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=300, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_epochs = args.epoch

    log = 'num_layer=' + str(num_layers) + \
    '_window_size=' + str(window_size) + \
    '_hidden=' + str(hidden_size) + \
    '_dataset=' + args.dataset + \
    '_epoch='+str(args.epoch)
    log = log + '_' + args.model

    if args.dataset == 'hd':
        seq_dataset = generate_hd('hdfs_train')
        num_classes = 28
    elif args.dataset == 'bgl':
        seq_dataset = generate_bgl('bgl', window_size)
        num_classes = 377

    if args.model == 'dl':
        model = DL(input_size, hidden_size, num_layers, num_classes)
    elif args.model == 'att':
        model = Att(input_size, hidden_size, num_layers, num_classes)
    elif args.model == 'trainable_att':
        model = Trainable_Att(input_size, hidden_size, num_layers, num_classes)

    model = model.to(device)

    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    writer = SummaryWriter(log_dir='log/' + log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        tbar = tqdm(dataloader)
        for step, (seq, label) in enumerate(tbar):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            tbar.set_description('Train loss: %.3f' % (train_loss / (step + 1)))

        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')
