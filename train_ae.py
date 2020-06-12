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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(name+'/window_20future_0/normal.txt', 'r') as f_len:
        file_len = len(f_len.readlines())
    with open(name+'/window_20future_0/normal.txt', 'r') as f:
        for line in f.readlines():
            if num_sessions < int(file_len *0.5):
                num_sessions += 1
                line = tuple(map(lambda n: n, map(int, line.strip().split())))
                inputs.append(line[:20])
            else:
                break

    print('Number of sessions({}): {}'.format(name, num_sessions))
    outputs = inputs
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset

class AE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys, seq_len):
        super(AE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, hidden_state = self.encoder(x, (h0, c0))
        # result = []
        # for i in range(self.seq_len):
        out , hidden_state = self.decoder(out, hidden_state)
        out = self.fc(F.log_softmax(out))
        return out
            # out_i = F.log_softmax(out)
            # result.append(self.fc(out_i))
            
        # return result



if __name__ == '__main__':

    # Hyperparameters
    num_classes = 1834
    num_epochs = 50
    batch_size = 4096
    input_size = 1
    model_dir = 'model'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=20, type=int)
    parser.add_argument('-ratio', default=0.1, type=float)
    parser.add_argument('-epoch', default=100, type=int)

    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    ratio = args.ratio
    num_epochs = args.epoch

    log = 'window_size='+str(window_size) + '_ratio=' + str(ratio) + '_epoch=' + str(num_epochs)
    log = log + '_ae' 


    model = AE(input_size, hidden_size, num_layers, num_classes, window_size)
    model = model.to(device)

    seq_dataset = generate('bgl')
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
            label = label.to(device)
            output = model(seq)
            loss = 0

            for i in range(window_size):
                loss += criterion(output[:,i,:], label[:,i])
            # loss = 0
            # for i in range(window_size):
            #     loss += criterion(output[i], label.transpose(0,1)[i])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
            tbar.set_description('Train loss: %.3f' % (train_loss / (step + 1)))
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')
    writer.close()
    print('Finished Training')
