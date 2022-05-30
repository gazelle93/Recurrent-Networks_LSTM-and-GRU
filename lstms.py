import torch
import torch.nn as nn

# RNN cell
class rnncell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(rnncell, self).__init__()
        # update gate weight & bias
        self.update_gate_input = nn.Linear(input_dim, output_dim)
        self.update_gate_hidden = nn.Linear(input_dim, output_dim)

        # tanh
        self.tanh = nn.Tanh()

    def forward(self, input_embedding, previous_hidden_states, previous_cell_states=None):
        output = self.tanh(self.update_gate_input(input_embedding)+self.update_gate_hidden(previous_hidden_states))

        return output, None

# LSTM cell
class lstmcell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(lstmcell, self).__init__()
        # input gate weight & bias
        self.input_gate_input = nn.Linear(input_dim, output_dim)
        self.input_gate_hidden = nn.Linear(input_dim, output_dim)

        # forget gate weight & bias
        self.forget_gate_input = nn.Linear(input_dim, output_dim)
        self.forget_gate_hidden = nn.Linear(input_dim, output_dim)

        # cell gate weight & bias
        self.cell_gate_input = nn.Linear(input_dim, output_dim)
        self.cell_gate_hidden = nn.Linear(input_dim, output_dim)

        # output gate weight & bias
        self.output_gate_input = nn.Linear(input_dim, output_dim)
        self.output_gate_hidden = nn.Linear(input_dim, output_dim)

        # sigmoid
        self.sigmoid = nn.Sigmoid()
        # tanh
        self.tanh = nn.Tanh()

    def forward(self, input_embedding, previous_hidden_states, previous_cell_states):
        _input = self.sigmoid(self.input_gate_input(input_embedding)+self.input_gate_hidden(previous_hidden_states))
        _forget = self.sigmoid(self.forget_gate_input(input_embedding)+self.forget_gate_hidden(previous_hidden_states))
        _cell = self.tanh(self.cell_gate_input(input_embedding)+self.cell_gate_hidden(previous_hidden_states))
        _output = self.sigmoid(self.output_gate_input(input_embedding)+self.output_gate_hidden(previous_hidden_states))

        current_cell = _forget * previous_cell_states + _input * _cell
        output = _output * self.tanh(current_cell)

        return output, current_cell

# GRU cell
class grucell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(grucell, self).__init__()
        # reset gate weight & bias
        self.reset_gate_input = nn.Linear(input_dim, output_dim)
        self.reset_gate_hidden = nn.Linear(input_dim, output_dim)

        # update gate weight & bias
        self.update_gate_input = nn.Linear(input_dim, output_dim)
        self.update_gate_hidden = nn.Linear(input_dim, output_dim)

        # output gate weight & bias
        self.new_gate_input = nn.Linear(input_dim, output_dim)
        self.new_gate_hidden = nn.Linear(input_dim, output_dim)

        # sigmoid
        self.sigmoid = nn.Sigmoid()
        # tanh
        self.tanh = nn.Tanh()

    def forward(self, input_embedding, previous_hidden_states, previous_cell_states=None):
        _reset = self.sigmoid(self.reset_gate_input(input_embedding)+self.reset_gate_hidden(previous_hidden_states))
        _update = self.sigmoid(self.update_gate_input(input_embedding)+self.update_gate_hidden(previous_hidden_states))
        current_cell = self.tanh(self.new_gate_input(input_embedding)+_reset*self.update_gate_hidden(previous_hidden_states))
        output = (1-_update) * current_cell + _update * previous_hidden_states

        return output, current_cell


class lstmlayer(nn.Module):
    def __init__(self, input_dim, output_dim, reverse=False, drouput_rate=0.1, sequential_model="lstm"):
        super(lstmlayer, self).__init__()
        self.reverse = reverse

        if sequential_model == "lstm":
            self.forwardlayer = lstmcell(input_dim, output_dim)
            if reverse:
                self.reverselayer = lstmcell(input_dim, output_dim)

        elif sequential_model == "gru":
            self.forwardlayer = grucell(input_dim, output_dim)
            if reverse:
                self.reverselayer = grucell(input_dim, output_dim)

        self.dropout = nn.Dropout(p=drouput_rate)

    def forward(self, input_embedding, previous_cell_states=None, hidden_states=None, is_dropout=True):
        output_list = torch.zeros((input_embedding.size()))
        cell_list = torch.zeros((input_embedding.size()))

        if hidden_states == None:
            previous_cell_states = torch.zeros(input_embedding[:,0,:].size())

        if hidden_states == None:
            hidden_states = torch.zeros(input_embedding[:,0,:].size())


        if self.reverse:
            output, cell = self.reverselayer(input_embedding=input_embedding[:,-1,:],
                                             previous_hidden_states=hidden_states,
                                             previous_cell_states=previous_cell_states)

            output_list[:,0,:] = output
            cell_list[:,0,:] = cell

            for idx in range(1, input_embedding.size()[1]):
                output, cell = self.reverselayer(input_embedding[:,input_embedding.size()[1]-idx-1,:],
                                                 previous_hidden_states=output,
                                                 previous_cell_states=cell)
                output_list[:,idx,:] = output
                cell_list[:,idx,:] = cell

        else:
            output, cell = self.forwardlayer(input_embedding=input_embedding[:,0,:],
                                             previous_hidden_states=hidden_states,
                                             previous_cell_states=previous_cell_states)

            output_list[:,0,:] = output
            cell_list[:,0,:] = cell

            for idx in range(1, input_embedding.size()[1]):
                output, cell = self.forwardlayer(input_embedding[:,idx,:],
                                                 previous_hidden_states=output,
                                                 previous_cell_states=cell)
                output_list[:,idx,:] = output
                cell_list[:,idx,:] = cell

        if is_dropout:
            output_list = self.dropout(output_list)

        return output_list, cell_list

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, bidirection=False, sequential_model="lstm"):
        super(LSTM, self).__init__()
        self.bidirection = bidirection

        self.forwardlstmlayer = lstmlayer(input_dim, output_dim, sequential_model=sequential_model)

        if bidirection:
            self.reverselstmlayer = lstmlayer(input_dim, output_dim, bidirection, sequential_model=sequential_model)

    def forward(self, input_embedding):
        forward_output, forward_cell = self.forwardlstmlayer(input_embedding=input_embedding, hidden_states=None, previous_cell_states=None, is_dropout=False)

        if self.bidirection:
            reverse_output, reverse_cell = self.reverselstmlayer(input_embedding=input_embedding, hidden_states=None, previous_cell_states=None, is_dropout=False)
            output = torch.cat((forward_output, reverse_output), dim=2)
            cell = torch.cat((forward_cell, reverse_cell), dim=2)
            return output, cell

        else:
            return forward_output, forward_cell
