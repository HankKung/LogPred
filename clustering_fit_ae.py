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
from ae.ae import AE, KMEANS

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_bgl(window_size):
    num_sessions = 0
    inputs = []
    outputs = []
    num_keys = set()
    with open('bgl/window_'+str(window_size)+'future_0/normal_train.txt', 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_'+str(window_size)+'future_0/normal_train.txt', 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line)
            outputs.append(line)
            for key in line:
                num_keys.add(key)
    print('Number of sessions: {}'.format(num_sessions))
    print('number of keys:{}'.format(len(num_keys)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset

def generate_hdfs(window_size):
    num_sessions = 0
    inputs = []
    outputs = []
    with open('data/hdfs_train', 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i:i + window_size])
    print('Number of sessions: {}'.format(num_sessions))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


if __name__ == '__main__':

    # Hyperparameters
    batch_size = 2048
    input_size = 1
    model_dir = 'model'
    
    parser = argparse.ArgumentParser()
    # ae
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=20, type=int)
    parser.add_argument('-window_size', default=20, type=int)

    # training
    parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=150, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-caption', type=str, default='')

    # k-means
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-iter', default=10, type=int)

    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    latent_length = args.latent_length
    window_size = args.window_size
    num_epochs = args.epoch
    k = args.k
    max_iter = args.iter 

    if args.dataset == 'hd':
        seq_dataset = generate_hdfs(window_size)
        num_classes = 28
        # for -1 padding during testing
        num_classes +=1
    elif args.dataset == 'bgl':
        seq_dataset = generate_bgl(window_size)
        num_classes = 1834
    
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model_path = 'model/'
    log = model_path + \
    'dataset=' + args.dataset + \
    '_window_size=' + str(window_size) + \
    '_hidden_size=' + str(hidden_size) + \
    '_latent_length=' + str(latent_length) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs)
    log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
    log = log + '_ae' + args.caption + '.pt' 
    print('store model at:')
    print(log)


    model = AE(input_size, hidden_size, latent_length, num_layers, num_classes, window_size)
    model = model.to(device)
    model.load_state_dict(torch.load(log))
    model.eval()


    k_means = KMEANS(n_clusters=k, max_iter=max_iter, verbose=True, device=device)

    k_means_path = log[:-3] + '_' +str(k) + '/'
    if not os.path.isdir(k_means_path):
        os.makedirs(k_means_path)

    all_vector = torch.empty((0, latent_length)).to(device)
    total_step = len(dataloader)
    tbar = tqdm(dataloader)
    for step, (seq, label) in enumerate(tbar):
        # Forward pass
        seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
        latent_vector = model.get_latent(seq)
        all_vector = torch.cat([all_vector, latent_vector], (0))

    store_vector = all_vector.cpu().data.numpy()
    np.save(k_means_path + 'latent_vector', store_vector)

    k_means.fit(all_vector)

    for i, center in enumerate(k_means.centers):
        center_store = center.cpu().data.numpy()
        np.save(k_means_path + 'center_' + str(i), center_store)
        # print(center_store)


