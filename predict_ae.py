import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from tqdm import tqdm

# Device configuration
device = torch.device("cuda")


def generate_bgl(name, window_size):
    num_sessions = 0
    inputs = []
    with open('bgl/window_'+str(window_size)+'future_0/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_'+str(window_size)+'future_0/' + name, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line)
            num_sessions += 1

    print('Number of sessions({}): {}'.format(name, num_sessions))
    return inputs

def generate_hdfs(name, window_size):
    hdfs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            ### pad -1 if the sequence len is less than the window_size
            line = line + [-1] * (window_size + 1 - len(line))
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                hdfs.append(tuple(seq))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs

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
        h_e = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_e = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        h_d = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_d = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.encoder(x, (h_e, c_e))
        out , _ = self.decoder(out, (h_d, c_d))
        out = self.fc(out)
        return out


if __name__ == '__main__':

    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'])
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-window_size', default=20, type=int)
    parser.add_argument('-epoch', default=100, type=int)
    parser.add_argument('-error_threshold', default=0.1, type=float)
    parser.add_argument('-capture', type=str, default='')
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_epochs = args.epoch
    threshold = args.error_threshold
    input_size = 1

    log = 'model/' + \
    'dataset=' + args.dataset + \
    '_window_size=' + str(window_size) + \
    '_hidden_size=' + str(hidden_size) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs)
    log = log + '_ae' + args.capture + '.pt' 

    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'hd':
        test_normal_loader = generate_hdfs('hdfs_test_normal', window_size)
        test_abnormal_loader = generate_hdfs('hdfs_test_abnormal', window_size)
        num_classes = 28
    elif args.dataset == 'bgl':
        test_normal_loader = generate('normal_test.txt', window_size)
        test_abnormal_loader = generate('abnormal_test.txt', window_size)
        num_classes = 1834
    num_classes +=1
    len_normal = len(test_normal_loader)
    len_abnormal = len(test_abnormal_loader)

    model = AE(input_size, hidden_size, num_layers, num_classes, window_size).to(device)
    model.load_state_dict(torch.load(log))
    model.eval()

    TP = 0
    FP = 0
    # Test the model
    tbar = tqdm(test_normal_loader)
    with torch.no_grad():
        normal_error = 0.0
        for index, line in enumerate(tbar):
            tbar.set_description('')

            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(line).unsqueeze(0).to(device)
            output = model(seq)
            loss = 0
            for i in range(window_size):
                loss += criterion(output[:,i,:], label[:,i]).item()
            if loss > threshold:
                FP += 1
            normal_error +=loss

    tbar = tqdm(test_abnormal_loader)
    with torch.no_grad():
        abnormal_error = 0.0
        for index, line in enumerate(tbar):
            tbar.set_description('')

            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(line).unsqueeze(0).to(device)
            output = model(seq)
            loss = 0
            for i in range(window_size):
                loss += criterion(output[:,i,:], label[:,i]).item()
            if loss > threshold:
                TP += 1
            abnormal_error += loss

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