import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from tqdm import tqdm

# Device configuration
device = torch.device("cuda")


def generate_normal(name):
    num_sessions = 0
    inputs = []
    with open('bgl/window_20future_0/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_20future_0/' + name, 'r') as f:
        for line in f.readlines():
            if num_sessions >= int(file_len *0.5):
                num_sessions += 1
                line = tuple(map(lambda n: n, map(int, line.strip().split())))
                inputs.append(line[:20])
            else:
                break

    print('Number of sessions({}): {}'.format(name, num_sessions))
    return inputs

def generate_abnormal(name):
    num_sessions = 0
    inputs = []
    with open('bgl/window_20future_0/' + name, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open('bgl/window_20future_0/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            inputs.append(line[:20])


    print('Number of sessions({}): {}'.format(name, num_sessions))
    return inputs

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
    input_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    parser.add_argument('-ratio', default=0.1, type=float)
    parser.add_argument('-epoch', default=100, type=int)
    parser.add_argument('-error', default=0.1, type=float)

    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates


    model = AE(input_size, hidden_size, num_layers, num_classes, window_size)

    'window_size='+str(window_size) + '_ratio=' + str(ratio) + '_epoch=' + str(num_epochs)
    log = log + '_ae.pt' 

    model.load_state_dict(torch.load(log))
    model.eval()
    test_normal_loader = generate_normal('normal.txt')
    test_abnormal_loader = generate_abnormal('abnormal.txt')
    TP = 0
    FP = 0
    # Test the model
    tbar = tqdm(test_normal_loader)
    with torch.no_grad():
        for index, line in enumerate(tbar):
            tbar.set_description('')

            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(line).view(-1).to(device)
            output = model(seq)
            loss = 0
            for i in range(window_size):
                loss += criterion(output[:,i,:], label[:,i]).item()

            if loss > error:
                FP += 1
                break

    tbar = tqdm(test_abnormal_loader)
    with torch.no_grad():
        for index, line in enumerate(tbar):
            tbar.set_description('')

            seq = torch.tensor(line, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(line).view(-1).to(device)
            output = model(seq)
            loss = 0
            for i in range(window_size):
                loss += criterion(output[:,i,:], label[:,i]).item()
            if loss > error:
                TP += 1
                break
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')