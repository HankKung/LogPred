import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DL(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(DL, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    def get_latent(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return torch.sum(out, 1)

class Att(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Att, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

        size = 0
        for p in self.parameters():
          size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))
        final_hidden_state = final_hidden_state[-1]
        out = self.attention_net(out, final_hidden_state)
        out = self.fc(out)
        return out

    def attention_net(self, lstm_output, final_state):

        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        
        Arguments
        ---------
        
        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM
        
        ---------
        
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.
                  
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                      
        """
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

class Trainable_Att(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Trainable_Att, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)
        self.att_w = nn.Parameter(torch.tensor(torch.randn(hidden_size).cuda()))
        size = 0
        for p in self.parameters():
          size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))
        final_hidden_state = final_hidden_state[-1]
        att_out = self.attention_net(out, final_hidden_state)
        out = self.fc(att_out)
        return out

    def attention_net(self, lstm_output, final_state):
        batch_size = lstm_output.shape[0]
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(torch.tanh(lstm_output), self.att_w.expand(batch_size, self.att_w.shape[0]).unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        new_hidden_state = torch.tanh(new_hidden_state)
        return new_hidden_state