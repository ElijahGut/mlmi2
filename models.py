import torch.nn as nn

class BiLSTM(nn.Module):
        def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout):
                super().__init__()
                self.dropout=dropout
                self.num_layers = num_layers
                if self.num_layers == 1:
                       self.num_layers = num_layers 
                       self.lstm = nn.LSTM(in_dims, hidden_dims, self.num_layers, bidirectional=True)
                       self.proj = nn.Linear(hidden_dims * 2, out_dims)
                else:
                       self.lstm = nn.LSTM(in_dims, hidden_dims, self.num_layers, bidirectional=True, dropout=self.dropout)
                       self.proj = nn.Linear(hidden_dims * 2, out_dims)

        def forward(self, feat):
                hidden, _ = self.lstm(feat)
                output = self.proj(hidden)
                if self.num_layers == 1:
                        d = nn.Dropout(p=self.dropout)
                        output = d(output)
                return output
