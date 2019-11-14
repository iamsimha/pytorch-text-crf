import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, batch_first, dropout, bidirectional, device="cpu"):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                        num_layers=num_layers, batch_first=batch_first, dropout=dropout,
                        bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.device = device

    def init_hidden(self, batch_size):
        return (torch.FloatTensor(self.num_layers * self.num_directions,
                                  batch_size, self.hidden_dim).fill_(0).to(device=self.device),
                torch.FloatTensor(self.num_layers * self.num_directions,
                                  batch_size, self.hidden_dim).fill_(0).to(device=self.device))

    def init_cell(self, batch_size):
        return torch.FloatTensor(self.num_layers * self.num_directions,
                                 batch_size, self.hidden_dim).fill_(0).to(device=self.device)

    def forward(self, batch, batch_size, hidden_state=None):
        if hidden_state is None:
            self.hidden = self.init_hidden(batch_size)
        else:
            self.hidden = (hidden_state, self.init_cell(batch_size))
        output, (h_n, c_n) = self.lstm(batch, self.hidden)
        return output, (h_n, c_n)

    def get_output_dim(self):
        return self.num_directions * self.hidden_dim
