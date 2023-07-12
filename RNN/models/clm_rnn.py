import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import timeit
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def criterion_by_masked(recon: torch.Tensor, target: torch.Tensor, mask_tensor: torch.Tensor):

    loss = F.cross_entropy(recon, target.long(), reduction='none')
    loss = loss * mask_tensor
    return torch.sum(loss)

class NNLM(nn.Module):

    # Define model parameters
    def __init__(self, **args):
        super(NNLM, self).__init__()

        # Model parameters
        size_of_one_hot = args['size_of_one_hot']
        units_of_rnn = args['units_of_rnn']
        layers_of_rnn = args['layers_of_rnn']
        dropout_rate = args['dropout_rate']
        self.param = args

        # Model layers
        self.lstm = nn.LSTM(input_size=size_of_one_hot, hidden_size=units_of_rnn, num_layers=layers_of_rnn,
                            dropout=dropout_rate, batch_first=True)
        self.linear = nn.Linear(units_of_rnn, size_of_one_hot)

    # Define initial hidden and cell states
    def init_states(self, batch_size):
        layers_of_rnn = self.param['layers_of_rnn']
        units_of_rnn = self.param['units_of_rnn']

        hidden = [torch.zeros(layers_of_rnn, batch_size, units_of_rnn, device=device),
                  torch.zeros(layers_of_rnn, batch_size, units_of_rnn, device=device)]

        # Initialize forget gate bias to 1
        # for names in self.lstm._all_weights:
        #     for name in filter(lambda n: "bias" in n, names):
        #         bias = getattr(self.lstm, name)
        #         n = bias.size(0)
        #         start, end = n // 4, n // 2
        #
        #         nn.init.constant_(bias.data[start:end], 1.0)

        return hidden

    # Define forward propagation
    def forward(self, inp, hidden):
        # LSTM
        output, hidden = self.lstm(inp, hidden)  # output [batch_size, seq_length, units_of_rnn]

        # reshape output
        output = output.reshape(-1, self.param['units_of_rnn'])  # output [batch_size * seq_length, units_of_rnn]

        # Linear Layer
        output = self.linear(output)  # output [batch_size * seq_length, one_oht_size]

        return output, hidden


def train(model, optimizer, data_loader, epochs, name):
    records = []
    model.train()

    for epoch in (bar := tqdm(range(1, epochs + 1), unit='epoch')):
        train_loss = 0
        for batch_idx, data in enumerate(data_loader):

            model.train()  # 启用 BatchNormalization 和 Dropout

            oh_pre, lb, l_of_len = data

            oh_pre = oh_pre.to(device)  # [batch_size, length-1, one_hot_size]
            lb = lb.long().to(device)
            l_of_len = l_of_len.byte().to(device)

            lb = lb.reshape(-1)  # lb [batch_size * seq_length]
            l_of_len = l_of_len.reshape(-1)   # [batch_size * seq_length]

            hidden = model.init_states(oh_pre.size(0))

            optimizer.zero_grad()
            recon, _ = model(oh_pre, hidden)  # recon [batch_size * seq_length, one_hot_size]

            recon = recon.reshape(-1, 26)
            lb = torch.flatten(lb)
            l_of_len = torch.flatten(l_of_len)
            loss = criterion_by_masked(recon, lb, l_of_len)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss

        train_loss = train_loss / len(data_loader.dataset)
        record = [epoch, train_loss.item()]
        records.append(record)
        bar.set_description(f'Loss: {train_loss}')

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch}
    torch.save(state, '{}.pth'.format(name))

    return records
