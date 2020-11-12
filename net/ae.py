import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
    def __init__(self, input_size, hidden_size, latent, num_layers, num_keys, seq_len, dropout_rate=0.0):
        super(AE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.compress_e = nn.Linear(hidden_size, latent)
        self.compress_d = nn.Linear(latent, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        nn.init.xavier_uniform_(self.compress_e.weight)
        nn.init.xavier_uniform_(self.compress_d.weight)
        size = 0

        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, x):
        h_e = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_e = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        decoder_inputs = torch.zeros(self.seq_len, x.shape[0], 1, requires_grad=False).type(torch.FloatTensor).cuda()
        
        h_d = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_d = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (_, _) = self.encoder(x, (h_e, c_e))
        # out = torch.sum(out, dim=0)
        out = self.compress_e(out)
        out = self.relu(out)
        out = self.compress_d(out)
        out , _ = self.decoder(out, (h_d, c_d))

        out = self.fc(out)
        if out.shape[0] != 1:
            out = self.dropout(out)
        return out

    def get_latent(self, x):
        h_e = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_e = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        h_d = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_d = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (_, _) = self.encoder(x, (h_e, c_e))
        out = self.compress_e(out)
        return out

class AE_Prediction(nn.Module):
    def __init__(self, input_size, hidden_size, latent, num_layers, num_keys, seq_len, dropout_rate=0.0):
        super(AE_Prediction, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.compress_e = nn.Linear(hidden_size, latent)
        self.compress_d = nn.Linear(latent, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        nn.init.xavier_uniform_(self.compress_e.weight)
        nn.init.xavier_uniform_(self.compress_d.weight)
        size = 0

        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, x):
        h_e = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_e = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        decoder_inputs = torch.zeros(self.seq_len, x.shape[0], 1, requires_grad=False).type(torch.FloatTensor).cuda()
        
        h_d = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_d = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (_, _) = self.encoder(x, (h_e, c_e))
        # out = torch.sum(out, dim=0)
        out = self.compress_e(out)
        out = self.relu(out)
        out = self.compress_d(out)
        out , _ = self.decoder(out, (h_d, c_d))

        out = self.fc(out)
        if out.shape[0] != 1:
            out = self.dropout(out)
        return out


# code modified from https://zhuanlan.zhihu.com/p/65331686 
class KMEANS:
    def __init__(self, n_clusters=10, max_iter=None, verbose=True, device=torch.device("cuda")):

        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            self.nearest_center(x)
            
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-4 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        self.representative_samples = torch.argmin(self.dists, (0))