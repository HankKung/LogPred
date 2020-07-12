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

class AE(nn.Module):
    def __init__(self, input_size, hidden_size, latent, num_layers, num_keys, seq_len):
        super(AE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_size, num_layers)
        self.compress_e = nn.Linear(hidden_size, latent)
        self.compress_d = nn.Linear(latent, hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_keys)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, x):
        batch_size = x.size(0)
        h_e = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_e = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        decoder_inputs = torch.zeros(self.seq_len, batch_size, 1, requires_grad=True).type(torch.FloatTensor).cuda()

        _, (h_e, _) = self.encoder(x, (h_e, c_e))
        h_e = h_e[-1,:,:]
        h_e = self.compress_e(h_e)
        h_e = self.relu(h_e)
        h_e = self.compress_d(h_e)
        h_e = torch.stack([h_e for _ in range(self.num_layers)])
        c_d = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out , _ = self.decoder(decoder_inputs, (h_e, c_d))

        out = out.permute(1,0,2)
        out = self.fc(out)
        return out

if __name__ == '__main__':

    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'])

    # model parameters
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=128, type=int)
    parser.add_argument('-latent_length', default=20, type=int)
    parser.add_argument('-window_size', default=20, type=int)

    # training parameters
    parser.add_argument('-epoch', default=100, type=int)
    parser.add_argument('-lr', default=0.001, type=float)

    parser.add_argument('-error_threshold', default=0.1, type=float)
    parser.add_argument('-caption', type=str, default='')
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    latent_length = args.latent_length
    window_size = args.window_size
    num_epochs = args.epoch
    threshold = args.error_threshold
    input_size = 1

    log = 'model/' + \
    'dataset=' + args.dataset + \
    '_window_size=' + str(window_size) + \
    '_hidden_size=' + str(hidden_size) + \
    '_latent_length=' + str(latent_length) + \
    '_num_layer=' + str(num_layers) + \
    '_epoch=' + str(num_epochs)
    log = log + '_lr=' + str(args.lr) if args.lr != 0.001 else log
    log = log + '_ae' + args.caption + '.pt' 

    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'hd':
        test_normal_loader = generate_hdfs('hdfs_test_normal', window_size)
        test_abnormal_loader = generate_hdfs('hdfs_test_abnormal', window_size)
        num_classes = 28
        num_classes +=1
    elif args.dataset == 'bgl':
        test_normal_loader = generate_bgl('normal_test.txt', window_size)
        test_abnormal_loader = generate_bgl('abnormal_test.txt', window_size)
        num_classes = 1834
    
    len_normal = len(test_normal_loader)
    len_abnormal = len(test_abnormal_loader)

    model = AE(input_size, hidden_size, latent_length, num_layers, num_classes, window_size).to(device)
    model.load_state_dict(torch.load(log))
    model.eval()

    TP = 0
    FP = 0
    # Test the model
    tbar = tqdm(test_normal_loader)
    with torch.no_grad():
        normal_error = 0.0
        for index, line in enumerate(tbar):

            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(line).to(device)
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