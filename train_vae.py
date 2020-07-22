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
from vae.vae import VRAE
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_bgl(name, window_size):
    num_sessions = 0
    inputs = []
    outputs = []
    num_keys = set()
    with open('bgl/window_'+str(window_size)+'future_0/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_'+str(window_size)+'future_0/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line)
            outputs.append(line)
            for key in line:
                num_keys.add(key)
    print(name)
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
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=20, type=int)
    parser.add_argument('-window_size', default=20, type=int)
    parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=150, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-dropout', default=0.0, type=float)
    parser.add_argument('-caption', type=str, default='')
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_epochs = args.epoch
    latent_length = args.latent_length
    dropout = args.dropout

    if args.dataset == 'hd':
        seq_dataset = generate_hdfs(window_size)
        num_classes = 28
        # for -1 padding during testing
        num_classes +=1
    elif args.dataset == 'bgl':
        seq_dataset = generate_bgl('normal_train.txt', window_size)
        val_dataset = generate_bgl('abnormal_test.txt', window_size)
        num_classes = 1848
    
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dataset = DataLoader(val_dataset, batch_size=4096, shuffle=False, pin_memory=True)

    log = 'dataset='+ str(args.dataset) + \
    '_window_size=' + str(window_size) + \
    '_hidden_size=' + str(hidden_size) + \
    '_latent_length=' + str(latent_length) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs) + \
    '_dropout=' + str(dropout)
    log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
    log = log + '_vae' + args.caption
    print('store model at:')
    print(log)
    writer = SummaryWriter(log_dir='log/' + log)


    model = VRAE(sequence_length=window_size,
            number_of_features=1,
            num_classes=num_classes,
            hidden_size=hidden_size,
            latent_length=latent_length,
            training=True,
            dropout_rate=dropout)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Train the model
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        train_loss = 0
        tbar = tqdm(dataloader)
        model.train()
        for step, (seq, label) in enumerate(tbar):
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            # label = label.to(device)
            # output = model(seq)
            
            loss, rec, kl = model.compute_loss(seq)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            tbar.set_description('Train loss: %.3f' % (train_loss / (step + 1)))
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)

        # if epoch % 20 == 0:
        #     model.eval()
        #     val_loss = 0.0
        #     tbar_val = tqdm(val_dataset)
        #     for step, (seq, label) in enumerate(tbar_val):
        #         # Forward pass
        #         seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
        #         with torch.no_grad():
        #             loss, _, _ = model.compute_loss(seq)

        #         val_loss += loss.data.cpu()
        #         tbar.set_description('val loss: %.3f' % (val_loss / (step + 1)))

    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')