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
from sklearn.manifold import TSNE
from pathlib import Path
from net.dl import *


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_bgl(window_size):
    num_sessions = 0
    inputs = set()
    outputs = set()
    num_keys = set()
    with open('bgl/window_'+str(window_size)+'future_0/normal_train.txt', 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_'+str(window_size)+'future_0/normal_train.txt', 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.add(line)
            outputs.add(line)
            for key in line:
                num_keys.add(key)
    print('Number of sessions: {}'.format(num_sessions))
    print('number of keys:{}'.format(len(num_keys)))
    inputs = list(inputs)
    outputs = list(outputs)
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs, dtype=torch.float))
    return dataset

def generate_hdfs(window_size):
    num_sessions = 0
    inputs = set()
    outputs = set()
    with open('data/hdfs_train', 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.add(line[i:i + window_size])
                outputs.add(line[i:i + window_size])
    print('Number of sessions: {}'.format(num_sessions))
    inputs = list(inputs)
    outputs = list(outputs)
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs, dtype=torch.float))
    return dataset


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
    parser.add_argument('-iter', default=10, type=int)

    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    latent_length = args.latent_length
    window_size = args.window_size
    num_epochs = args.epoch
    dropout = args.dropout
    k = args.k
    max_iter = args.iter 

    if args.dataset == 'hd':
        seq_dataset = generate_hdfs(window_size)
        num_classes = 28
        # for -1 padding during testing
        if args.model != 'dl':
            num_classes +=1
    elif args.dataset == 'bgl':
        seq_dataset = generate_bgl(window_size)
        num_classes = 1848
    
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

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

    # t-SNE
    # store_vector = all_vector.cpu().data.numpy()
    # embedded = Path(k_means_path + 'embedded_vector.npy')
    # if not embedded.is_file():
    #     z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(store_vector)
    #     np.save(k_means_path + 'embedded_vector', z_run_tsne)
    # else:
    #     z_run_tsne = np.load('embedded_vector.npy')
    # k_means.fit(torch.from_numpy(z_run_tsne).cuda())

    k_means.fit(all_vector)

    for i, center in enumerate(k_means.centers):
        center_store = center.cpu().data.numpy()
        np.save(k_means_path + 'center_' + str(i), center_store)
        # print(center_store)


