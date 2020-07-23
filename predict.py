import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from tqdm import tqdm
from deeplog.model import *

# Device configuration
device = torch.device("cuda")


def generate_hd(name, window_size):
    hdfs = set()
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs

def generate_bgl(split, window_size):
    num_sessions = 0
    bgl = []
    ### future one: DeepLog framework predict the next one log
    with open(name+'bgl/window_' + str(window_size) +'future_1/'+ split, 'r') as f_len:
        file_len = len(f_len.readlines())
    with open(name+'bgl/window_'+ str(window_size) + 'future_1/'+ split, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            bgl.append(line)
            
    print('Number of sessions({}): {}'.format(name, num_sessions))
    return bgl


if __name__ == '__main__':

    # Hyperparameters
    input_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    parser.add_argument('-model', type=str, default='dl', \
        choices=['dl', 'att', 'trainable_att'])
    parser.add_argument('-dataset', type=str, default='hd', \
        choices=['hd', 'bgl'])
    parser.add_argument('-epoch', default=300, type=int)

    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    log = 'model/num_layer=' + str(num_layers) + \
    '_window_size=' + str(window_size) + \
    '_hidden=' + str(hidden_size) + \
    '_dataset=' + args.dataset +\
    '_epoch='+str(args.epoch)
    log = log + '_' + args.model
    log = log + '.pt'
    print('retrieve model from: ', log)

    if args.dataset == 'hd':
        test_normal_loader = generate_hd('hdfs_test_normal', window_size)
        test_abnormal_loader = generate_hd('hdfs_test_abnormal', window_size)
        num_classes = 28
    elif args.dataset == 'bgl':
        test_normal_loader = generate_bgl('normal_test.txt', window_size)
        test_abnormal_loader = generate_bgl('abnormal_test.txt', window_size)
        num_classes = 1848

    if args.model == 'dl':
        model = DL(input_size, hidden_size, num_layers, num_classes)
    elif args.model == 'att':
        model = Att(input_size, hidden_size, num_layers, num_classes)
    elif args.model == 'trainable_att':
        model = Trainable_Att(input_size, hidden_size, num_layers, num_classes)
    model = model.to(device)

    model.load_state_dict(torch.load(log))
    model.eval()

    TP = 0
    FP = 0
    # Test the model
    tbar = tqdm(test_normal_loader)
    if args.dataset == 'hd':
        with torch.no_grad():
            for index, line in enumerate(tbar):
                tbar.set_description('')
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        FP += 1
                        break
    elif args.dataset == 'bgl':
        with torch.no_grad():
            for index, line in enumerate(tbar):
                tbar.set_description('')
                seq = line[:window_size]
                label = line[-1]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1


    if args.dataset == 'hd':
        tbar = tqdm(test_abnormal_loader)
        with torch.no_grad():
            for index, line in enumerate(tbar):
                tbar.set_description('')
                for i in range(len(line) - window_size):
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    if label not in predicted:
                        TP += 1
                        break
    elif args.dataset == 'bgl':
        tbar = tqdm(test_abnormal_loader)
        with torch.no_grad():
            for index, line in enumerate(tbar):
                tbar.set_description('')
                seq = line[:window_size]
                label = line[-1]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1

    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')
