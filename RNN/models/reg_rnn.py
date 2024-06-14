import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNNPredictor(nn.Module):

    # Define model parameters
    def __init__(self, **args):
        super(RNNPredictor, self).__init__()

        # Model parameters
        size_of_one_hot = args['size_of_one_hot']
        units_of_rnn = args['units_of_rnn']
        layers_of_rnn = args['layers_of_rnn']
        units_of_nn = args['units_of_nn']
        dropout_rate = args['dropout_rate']
        self.param = args

        # Model layers
        self.lstm = nn.LSTM(input_size=size_of_one_hot, hidden_size=units_of_rnn, num_layers=layers_of_rnn,
                            dropout=dropout_rate, batch_first=True)
        self.output1 = nn.Linear(units_of_rnn, units_of_nn)
        self.output2 = nn.Linear(units_of_nn, 1)
        if args['activation'] == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    # Define initial hidden and cell states
    def init_states(self, batch_size):
        layers_of_rnn = self.param['layers_of_rnn']
        units_of_rnn = self.param['units_of_rnn']

        hidden = [torch.zeros(layers_of_rnn, batch_size, units_of_rnn, device=device),
                  torch.zeros(layers_of_rnn, batch_size, units_of_rnn, device=device)]

        return hidden

    # Define forward propagation
    def forward(self, inp, hidden, is_train=True):
        # LSTM
        output, hidden = self.lstm(inp, hidden)  # output [batch_size, seq_length, units_of_rnn]

        (h, cell) = hidden
        output = h[-1, :, :]

        # Linear Layer
        output = self.activation(self.output1(output))  # output [batch_size, one_oht_size]
        output = F.dropout(output, p=self.param['dropout_rate'])
        output = self.output2(output)  # output [batch_size]

        return output, hidden

def train(model, optimizer, train_loader, valid_x, valid_y, epochs, filename):

    #print('Start training')

    record_path = os.getcwd() + '/record/'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    records = []
    for epoch in tqdm(range(1, epochs + 1), unit='epoch', desc=f'Training {filename}'):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):

            model.train()

            (x_padded, lengths, y) = data
            x_packed = rnn_utils.pack_padded_sequence(x_padded, lengths, batch_first=True)  # 打包

            x_packed = x_packed.to(device)
            y = y.float().to(device)
            hidden = model.init_states(x_padded.size(0))

            optimizer.zero_grad()

            y_, _ = model(x_packed, hidden)

            loss = F.mse_loss(y_.squeeze(1), y, reduction='sum')
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss

        train_loss = train_loss / len(train_loader.dataset)

        y_predict = predict(model, valid_x)
        valid_y = torch.tensor(valid_y, dtype=torch.float).to(device)
        valid_loss = F.mse_loss(y_predict, valid_y, reduction='mean')

        record = [epoch, train_loss.item(), valid_loss.item()]
        records.append(record)

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epochs': epoch}
    torch.save(state, f'training_records/{filename}.pth')
    return records


def predict(model: torch.nn.Module, x: torch.tensor) -> torch.tensor:

    model.eval()

    y_predict = torch.zeros(len(x)).to(device)
    for index in range(len(x)):
        x_ = x[index].unsqueeze(0).to(device)
        y_, _ = model(x_, model.init_states(1), False)
        y_predict[index] = y_.to(device)

    return y_predict
