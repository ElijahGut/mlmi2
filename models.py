import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, num_layers, in_dims, hidden_dims, out_dims, dropout, is_bidir):
        super().__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_dims, hidden_dims, self.num_layers, bidirectional=False)
        if num_layers > 1:
            self.lstm = nn.LSTM(in_dims, hidden_dims, self.num_layers, bidirectional=False, dropout=dropout)
        self.proj = nn.Linear(hidden_dims * 1, out_dims)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        hidden = self.dropout_layer(hidden)
        output = self.proj(hidden)
        return output
